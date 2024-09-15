import pickle

from loguru import logger
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.losses import CircleLoss
from pytorch_metric_learning.miners import BatchHardMiner
from pytorch_metric_learning.samplers import MPerClassSampler
from torch import mean
from torch.optim import Adam
from torch_geometric.loader import DataLoader


class GNNTrainer:
    def __init__(
        self,
        model_cls,
        training_dataset_pickle: str,
        evaluation_dataset_pickle: str | None = None,
        learning_rate: float = 0.001,  # The "size" of the steps made when updating the GNN
        batch_size: int = 64,  # The number of examples fed into the neural network at once
        num_pos_pairs_in_batch: int = 2,  # The number of postiives per example
        num_training_epochs: int = 250,  # The number of runs through the data to use to train
        hidden_dimension_size: int = 64,  # Size of hidden dimension for the GNN
        output_dimension: int = 64,  # Size of the embdding/representation outputted by the GNN
        num_samples_per_epoch: int = 100000,  # The number of samples within each epoch
    ):
        logger.info("Initialising GNNTrainer")

        logger.info("Loading raw dataset")
        self.training_dataset_raw = pickle.load(open(training_dataset_pickle, "rb"))
        logger.info("Successfully loaded dataset")

        logger.info("Splitting data into data + func_names")
        self.training_data = [x[0] for x in self.training_dataset_raw]
        logger.info("Training data split complete")
        self.training_data_func_names = [x[1] for x in self.training_dataset_raw]
        logger.info("Training func names split complete")

        logger.info("Extracting labels from examples")
        self.labels = [example["y"] for example in self.training_data]
        logger.info("Label extraction complete")

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_pos_pairs_in_batch = num_pos_pairs_in_batch
        self.num_epochs = num_training_epochs

        self.sampler = MPerClassSampler(
            self.labels,
            self.num_pos_pairs_in_batch,
            self.batch_size,
            length_before_new_iter=num_samples_per_epoch,
        )

        self.model = model_cls(
            num_node_features=8,
            hidden_dimension=hidden_dimension_size,
            output_dimension=output_dimension,
        )

        self.miner = BatchHardMiner(distance=CosineSimilarity())
        self.loss_func = CircleLoss(m=0.25, gamma=256)
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)

        self.training_dataset = DataLoader(
            self.training_data, self.batch_size, sampler=self.sampler
        )

        if evaluation_dataset_pickle is not None:
            eval_dataset = pickle.load(open(evaluation_dataset_pickle, "rb"))
            self.eval_dataset = [x[0] for x in eval_dataset]

        logger.info("Trainer successfully initialised")

    def train(self):
        self.model.train()
        logger.info("Starting to train")
        for epoch in range(self.num_epochs):
            epoch_losses = []
            for i, data in enumerate(self.training_dataset):
                self.optimizer.zero_grad()
                out = self.model(data)

                hard_pairs = self.miner(out, data.y)
                loss = self.loss_func(out, data.y, hard_pairs)

                loss.backward()
                self.optimizer.step()
                epoch_losses.append(loss)

            logger.info(f"Epoch {epoch + 1} -> Avg Loss: {mean(loss)}")
