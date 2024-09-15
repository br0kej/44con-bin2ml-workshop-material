import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv
from torch_geometric.nn.pool import global_mean_pool


class GCN(torch.nn.Module):
    def __init__(
        self, num_node_features: int, hidden_dimension: int, output_dimension: int
    ):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dimension)
        self.conv2 = GCNConv(hidden_dimension, hidden_dimension)
        self.conv3 = GCNConv(hidden_dimension, output_dimension)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)

        return global_mean_pool(x, data.batch)


class GraphConvNet(torch.nn.Module):
    def __init__(
        self, num_node_features: int, hidden_dimension: int, output_dimension: int
    ):
        super().__init__()
        self.conv1 = GraphConv(num_node_features, hidden_dimension)
        self.conv2 = GraphConv(hidden_dimension, hidden_dimension)
        self.conv3 = GraphConv(hidden_dimension, output_dimension)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)

        return global_mean_pool(x, data.batch)
