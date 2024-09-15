import argparse
import json
import pickle
import random
from pathlib import Path
from typing import Tuple

import networkx as nx
from joblib import Parallel, delayed
from loguru import logger
from networkx.readwrite.json_graph import adjacency_graph
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from tqdm import tqdm


def load_graph_from_json(json_file):
    """
    Load a graph from a JSON file and return a networkx DiGraph object.
    """
    with open(json_file, "r") as f:
        data = json.load(f)

    if "adjacency" not in data:
        return None

    return adjacency_graph(data)


def calculate_betweenness_centrality(G):
    """
    Calculate the node betweenness centrality for a given networkx DiGraph.
    """
    return nx.betweenness_centrality(G)


def save_as_pickled_data(data_list, output_file):
    """
    Save a list of PyTorch Geometric Data objects as a pickled DataLoader.
    """
    with open(output_file, "wb") as f:
        pickle.dump(data_list, f)


def process_single_graph(filename: str):
    """
    Load a single graph and add the node features to it before converting it to a
    PyTorch Geometric data object
    """
    binary = Path(filename).parent.name.split("_")[-2]
    function_name = Path(filename).name.split("-")[-1][:-5]

    binary_func_idx = binary + "<->" + function_name

    try:
        G = load_graph_from_json(filename)
        if G is None:
            return None
        else:
            bb = nx.betweenness_centrality(G)
            nx.set_node_attributes(G, bb, "betweenness")

            G.graph["binary"] = binary
            G.graph["function_name"] = function_name

            node_attr_names = [
                "numCalls",
                "numTransfer",
                "numArith",
                "numIns",
                "numericConsts",
                "stringConsts",
                "numOffspring",
                "betweenness",
            ]
            pyg_data_tensor = from_networkx(G, group_node_attrs=node_attr_names)

            return pyg_data_tensor, binary_func_idx
    except Exception:
        print(f"Failed to process: {filename}")


def get_all_filenames(input_dir: str):
    """
    Get all filenames from a target directory
    """

    fps = [x for x in list(Path(input_dir).rglob("*.json")) if x.is_file()]

    if len(fps) == 0:
        raise IOError(f"Unable to find any valid filenames @ {input_dir}")
    return fps


def split_train_eval(filepaths: list[Path]) -> tuple[list[str], list[str]]:
    # Get unique binaries to split based on binary name
    unique_binaries = set()
    binary_to_path_mapping = dict()

    for filepath in filepaths:
        binary_name = str(filepath.parent).split("_")[-2]
        unique_binaries.add(binary_name)
        if binary_name not in binary_to_path_mapping:
            binary_to_path_mapping[binary_name] = []

        binary_to_path_mapping[binary_name].append(filepath)

    # Get 80% binaries for train and 20% for eval
    num_unique_binaries = len(unique_binaries)
    num_train_idx = int(num_unique_binaries * 0.8)

    # Convert set to list
    unique_binaries = list(unique_binaries)
    # Random sample
    random.shuffle(unique_binaries)

    # Split on train idx
    train_binaries = unique_binaries[:num_train_idx]
    test_binaries = unique_binaries[num_train_idx:]

    # Subset dict into list
    train_paths = [
        path for binary in train_binaries for path in binary_to_path_mapping[binary]
    ]
    test_paths = [
        path for binary in test_binaries for path in binary_to_path_mapping[binary]
    ]

    return train_paths, test_paths


def format_and_clean_up_data_objs(data_tensors: Tuple[Data, str]):
    # Remove any None data objects
    data_tensors = [x for x in data_tensors if x is not None]
    data_tensors = [
        (tensor, idx) for (tensor, idx) in data_tensors if tensor is not None
    ]

    # Process idx into numercial labels
    binary_func_idx_mapping = []

    for tensor, idx in data_tensors:
        if idx not in binary_func_idx_mapping:
            binary_func_idx_mapping.append(idx)

        idx_val = binary_func_idx_mapping.index(idx)
        tensor["y"] = idx_val

    return data_tensors


def process_e2e(input_dir: str, num_cpu: int, output_pkl_name: str):
    all_filenames = get_all_filenames(input_dir)
    data_tensors = Parallel(n_jobs=num_cpu)(
        delayed(process_single_graph)(filename) for filename in tqdm(all_filenames)
    )

    data_tensors = format_and_clean_up_data_objs(data_tensors)

    save_as_pickled_data(data_tensors, output_pkl_name)


def main():
    parser = argparse.ArgumentParser(description="Convert and pickle graphs")
    parser.add_argument(
        "-i",
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing JSON graph files",
    )
    parser.add_argument(
        "-n",
        "--num-cpu",
        type=int,
        default=-1,
        help="Set the number of CPU for joblib to use to process the graphs",
    )
    args = parser.parse_args()

    logger.info("Starting up!")
    logger.info(f"Getting file names from {args.input_dir}")
    process_e2e(args.input_dir, args.num_cpu, "full_dataset.pkl")


if __name__ == "__main__":
    main()
