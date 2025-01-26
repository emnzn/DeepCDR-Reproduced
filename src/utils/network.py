import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, 
    global_mean_pool, 
    BatchNorm as GraphBatchNorm,
    Sequential as GraphSequential
    )

class DeepCDR(nn.Module):
    def __init__(
        self,
        mode: str = "classification",
        output_dim: int = 100,
        dropout_prob: float = 0.1,
        ):
        super().__init__()

        valid_modes = ["classification", "regression"]
        assert mode in valid_modes, f"mode must be one of {valid_modes}"

        self.drug_net = DrugGCN(
            output_dim=output_dim, 
            dropout_prob=dropout_prob
            )
        
        self.gene_net = MLP(
            input_dim=697, 
            output_dim=output_dim, 
            dropout_prob=dropout_prob
            )
        
        self.methylation_net = MLP(
            input_dim=808, 
            output_dim=output_dim, 
            dropout_prob=dropout_prob
            )
        
        self.mutation_net = MutationConv1d(
            output_dim=output_dim, 
            dropout_prob=dropout_prob
            )

        self.projection = nn.Sequential(
            nn.Linear(output_dim*4, 300),
            nn.Tanh(),
            nn.Dropout(p=dropout_prob),
        )

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=30, kernel_size=150, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=30, out_channels=10, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),
            nn.Conv1d(in_channels=10, out_channels=5, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),
            nn.Dropout(p=dropout_prob),
            nn.Flatten(),
            nn.Dropout(p=0.2)
        )

        self.fc = nn.Linear(30, 2 if mode == "classification" else 1)

    def forward(
        self, 
        drug_graph, 
        gene_expression_features, 
        methylation_features, 
        mutation_features
        ):

        drug_embedding = self.drug_net(
            drug_graph.x, 
            drug_graph.edge_index, 
            drug_graph.batch
        )

        gene_expression_embedding = self.gene_net(
            gene_expression_features
        )

        methylation_embedding = self.methylation_net(
            methylation_features
        )

        mutation_embedding = self.mutation_net(
            mutation_features
        )

        combined_embedding = torch.cat([
            drug_embedding,
            gene_expression_embedding,
            methylation_embedding,
            mutation_embedding
        ], dim=-1)

        x = self.projection(combined_embedding)
        x = x.unsqueeze(1)
        x = self.conv(x)
        
        out = self.fc(x)

        return out


class DrugGCN(nn.Module):
    def __init__(
        self,
        input_dim: int = 75,
        hidden_dim: int = 256,
        num_hidden: int = 3,
        output_dim: int = 100,
        dropout_prob: float = 0.1
        ):
        super().__init__()

        self.embedding_layer = GraphSequential("x, edge_index", [
            (GCNConv(input_dim, hidden_dim), "x, edge_index -> x"),
            nn.ReLU(),
            GraphBatchNorm(hidden_dim),
            nn.Dropout(p=dropout_prob)
        ])

        hidden_layers = []

        for _ in range(num_hidden-2):
            hidden_layers.append((GCNConv(hidden_dim, hidden_dim), "x, edge_index -> x"))
            hidden_layers.append(nn.ReLU())
            hidden_layers.append(GraphBatchNorm(hidden_dim))
            hidden_layers.append(nn.Dropout(p=dropout_prob))
        
        self.hidden_layer = GraphSequential("x, edge_index", hidden_layers)

        self.output_layer = GraphSequential("x, edge_index", [
            (GCNConv(hidden_dim, output_dim), "x, edge_index -> x"),
            nn.ReLU(),
            GraphBatchNorm(output_dim),
            nn.Dropout(p=dropout_prob)
        ])
        
    def forward(self, x, edge_index, batch):
        x = self.embedding_layer(x, edge_index)
        x = self.hidden_layer(x, edge_index)
        x = self.output_layer(x, edge_index)

        embedding = global_mean_pool(x, batch)

        return embedding
    

class MutationConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        output_dim: int = 100,
        dropout_prob: float = 0.1
        ):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=50,
                kernel_size=700,
                stride=5
            ),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=5)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=50,
                out_channels=30,
                kernel_size=5,
                stride=2
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=10),
            nn.Flatten()
        )

        self.fc = nn.Sequential(
            nn.Linear(2010, output_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        embedding = self.fc(x)

        return embedding
    

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 256,
        output_dim: int = 100,
        dropout_prob: float = 0.1
        ):
        super().__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.Tanh(),
            nn.BatchNorm1d(embedding_dim),
            nn.Dropout(p=dropout_prob)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(embedding_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.fc1(x)
        embedding = self.fc2(x)

        return embedding