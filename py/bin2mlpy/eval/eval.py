import os
import pickle
import random
import time
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import List

import torch
import torch.nn as nn
from bin2mlpy.data_utils.convert_and_pickle import process_e2e
from loguru import logger
from torch_geometric.data import Data
from torchmetrics.retrieval import RetrievalMRR, RetrievalPrecision
from tqdm import tqdm


class SearchPoolGNNEvaluator:
    def __init__(
        self,
        model,
        eval_data: list[Data],
        search_pool_size: int = 100,
        num_search_pools: int = 50,
        validate: bool = False,
    ):
        logger.info("Initialising GNNEvaluator")
        self.model = model
        self.model.eval()

        self.eval_data = eval_data
        self.search_pool_size = search_pool_size
        self.num_search_pools = num_search_pools

        logger.info("Pre-sorting labels")
        self.labels = sorted([example["y"] for example in self.eval_data])
        self.eval_data = sorted(self.eval_data, key=lambda x: x.y)
        self.label_counter = Counter(self.labels)

        logger.info("Calculating singleton labels")
        singleton_labels = [k for k, v in self.label_counter.items() if v == 1]

        logger.info("Using singleton labels to clean data")
        for i, example in enumerate(self.eval_data):
            if int(example["y"]) in singleton_labels:
                del self.eval_data[i]
                del self.labels[i]

        self.label_counter = Counter(self.labels)
        self.eval_data_size = len(self.eval_data)

        if validate:
            for label, data in zip(self.labels, self.eval_data):
                assert label == data.y

        self.unique_labels = list(set(self.labels))

        logger.info(
            f"All processing complete. Unique Labels: {len(self.unique_labels)}"
        )

    def evaluate(self):
        pos_label_sample = random.choices(self.unique_labels, k=self.num_search_pools)

        scores = []
        targets = []
        indexes = []

        for i, label in tqdm(enumerate(pos_label_sample), total=self.num_search_pools):
            try:
                pairs = []
                offset_samples = random.sample(range(self.label_counter[label]), k=2)
                assert (
                    self.eval_data[self.labels.index(label) + offset_samples[0]]["y"]
                    == self.eval_data[self.labels.index(label) + offset_samples[1]]["y"]
                )
                pairs.append(
                    (
                        self.eval_data[self.labels.index(label) + offset_samples[0]],
                        self.eval_data[self.labels.index(label) + offset_samples[1]],
                    )
                )

                while len(pairs) != self.search_pool_size:
                    random_index = random.sample(range(self.eval_data_size), k=1)[0]
                    if (
                        self.eval_data[self.labels.index(label) + offset_samples[0]][
                            "y"
                        ]
                        != self.eval_data[random_index]["y"]
                    ):
                        pairs.append(
                            (
                                self.eval_data[
                                    self.labels.index(label) + offset_samples[0]
                                ],
                                self.eval_data[random_index],
                            )
                        )
                    else:
                        continue

                assert len(pairs) == self.search_pool_size

                for a, b in pairs:
                    with torch.inference_mode():
                        al = a["y"]
                        a = self.model.forward(a)

                        bl = b["y"]
                        b = self.model.forward(b)

                        sim = torch.cosine_similarity(a, b, dim=1).item()
                        scores.append(sim)
                        targets.append(True if bl == al else False)
                        indexes.append(i)
            except Exception:
                logger.warning(f"Failed to generate pairs for {al} label")

        logger.info("Calculating metrics!")
        indexes, scores, targets = (
            torch.LongTensor(indexes),
            torch.FloatTensor(scores),
            torch.LongTensor(targets),
        )

        mrr_10 = RetrievalMRR(top_k=10)
        recall_1 = RetrievalPrecision(top_k=1)

        logger.info(f"MRR @ 10: {mrr_10(scores, targets, indexes=indexes).item()}")
        logger.info(f"Recall @ 1: {recall_1(scores, targets, indexes=indexes).item()}")


class NDayGNNEvaluator:
    # This has been adapted from the source code released as part of FASER
    # https://github.com/br0kej/FASER/tree/main
    NETGEAR_VULNS = ["CMS_decrypt", "PKCS7_dataDecode", "MDC2_Update", "BN_bn2dec"]
    TPLINK_VULNS = [
        "CMS_decrypt",
        "PKCS7_dataDecode",
        "BN_bn2dec",
        "X509_NAME_oneline",
        "EVP_EncryptUpdate",
        "EVP_EncodeUpdate",
        "SRP_VBASE_get_by_user",
        "BN_dec2bn",
        "BN_hex2bn",
    ]

    def __init__(
        self,
        model: nn.Module,
        eval_data_dir: str,
    ):
        logger.info("Initializing N-Day Evaluator")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model
        self.model.to(self.device)
        self.model.eval()
        self.eval_data_dir = eval_data_dir

        logger.info("Searching for processed data")
        self.folders = [
            x for x in list(Path(self.eval_data_dir).rglob("*")) if x.is_dir()
        ]

        if len(self.folders) != 6:
            raise ValueError(
                f"The expected number of folders (i.e unique binaries) is 6 - Got {len(self.folders)}"
            )

        logger.info("Found the expected number of folders.")

        self.folder_name_versions = [
            "_".join(str(x).split("_")[1:4]) for x in self.folders
        ]

        logger.info("Starting to pre-process data")
        for folder, folder_name_version in zip(self.folders, self.folder_name_versions):
            expected_pkl_path = f"{folder_name_version}.pkl"
            if Path(expected_pkl_path).exists() is False:
                process_e2e(folder, 4, f"{folder_name_version}.pkl")
                logger.info(f"Successfully pre-processed data @ {expected_pkl_path}")
            else:
                logger.info(
                    f"Pre-existing pickle found for {expected_pkl_path}. Skipping..."
                )

        firmware_ids = ["NETGEAR", "TP-Link"]

        self.firmwares = [
            x
            for x in self.folder_name_versions
            if any(id in str(x) for id in firmware_ids)
        ]
        self.samples = [x for x in self.folder_name_versions if x not in self.firmwares]

        logger.info(f"The following firmware will be searched: {self.firmwares}")
        logger.info(
            f"The following samples will be used as the search pool database: {self.samples}]"
        )

        self.model_id = str(time.time_ns())[-8:]

    def rank(
        self,
        firmware_name: str,
        firmware_funcs_pkl: str,
        arch_query_funcs_pkl: str,
        firmware_vulns: List[str],
    ):
        firmware_arch = "arm32" if firmware_name == "netgear" else "mips32"
        sp_save_path = f"{self.model_id}-{firmware_name}-embedded.pickle"

        # Load firmware funcs
        firmware_funcs = pickle.load(open(f"{firmware_funcs_pkl}.pkl", "rb"))

        # Load target funcs
        target_funcs = pickle.load(open(f"{arch_query_funcs_pkl}.pkl", "rb"))

        target_func_arch = target_funcs[0][1].split("<->")[0]
        query_binary = " ".join(arch_query_funcs_pkl.split("_")[:2])
        logger.info("-----------------------------------")
        logger.info(
            f"{query_binary} ({target_func_arch}) -> {firmware_name} ({firmware_arch})"
        )
        logger.info("-----------------------------------")

        targets = []
        for func in target_funcs:
            if func[1].split("<->")[1][4:] in firmware_vulns:
                targets.append(func)

        # Generate Embeddings of search pool
        search_pool_embeddings = []
        if os.path.exists(sp_save_path):
            search_pool_embeddings = pickle.load(open(sp_save_path, "rb"))
        else:
            with torch.inference_mode():
                for datum in firmware_funcs:
                    embedded = self.model.forward(datum[0])
                    search_pool_embeddings.append((datum[1], embedded))
            with open(sp_save_path, "wb") as f:
                pickle.dump(search_pool_embeddings, f)

        # Generate Embeddings of targets
        target_embeddings = []
        with torch.inference_mode():
            for datum in targets:
                embedded = self.model.forward(datum[0])
                target_embeddings.append((datum[1], embedded))

        # Calculate Sims
        ranks = []
        sim_hits = []
        for name, target in target_embeddings:
            sims = []
            names = []

            for sp_name, sp_embed in search_pool_embeddings:
                sim = torch.cosine_similarity(target, sp_embed)
                sims.append(sim)
                names.append(sp_name.split("<->")[1])

            zipped = list(zip(sims, names))
            zipped.sort(reverse=True)

            for i, z in enumerate(zipped):
                function_name = name.split("<->")[1]
                if function_name == z[1]:
                    logger.info(
                        f"[+] Found {function_name} @ {i + 1} (Sim: {round(z[0].item(), 4)})"
                    )
                    ranks.append(i + 1)
                    sim_hits.append(round(z[0].item(), 4))

        logger.info("*****    Summary    *****")
        logger.info(f"Ranks: {ranks}")
        logger.info(f"Scores: {sim_hits}")
        logger.info(f"Mean Rank: {round(mean(ranks))} Median Rank: {median(ranks)}")

    def netgear_evaluate(self):
        netgear_path = [x for x in self.firmwares if "NETGEAR" in x][0]
        logger.info(
            "Starting the evaluation against Openssl taken from NETGEAR device..."
        )

        for target in self.samples:
            self.rank("netgear", netgear_path, target, self.NETGEAR_VULNS)

    def tplink_evaluate(self):
        logger.info(
            "Starting the evaluation against Openssl taken from TP-Link device..."
        )
        tplink_path = [x for x in self.firmwares if "TP-Link" in x][0]
        for target in self.samples:
            self.rank("tplink", tplink_path, target, self.TPLINK_VULNS)
