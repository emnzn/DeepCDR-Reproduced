import os
from typing import Dict

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch_geometric.data import Data as GraphData
from torch_geometric.loader import DataLoader as GraphDataLoader


class MultiOmicsDataset(Dataset):
    def __init__(
        self,
        table_path: str,
        drug_dir: str,
        cell_line_dir: str,
        mode: str = "classification"
        ):

        """
        The Dataset constructor.

        Parameters
        ----------
        table_path: str
            The path of the reference table with the following fields:
                - pair_id
                - label
                - ic50
                - cell_line_id
                - drug_id
                - cancer_type

        drug_dir: str
            The directory with drug features.
            Must be in the form:
                drugs/
                |--drug_id
                      |──drug-edge-list.npy
                      └──drug-feature.npy
        
        cell_line_dir: str
            The directory with cell line features.
            Must be in the form:
                cell-line/
                |--cell_line_id
                      |──gene-feature.npy
                      |──methylation-feature.npy
                      └──mutation-feature.npy

        mode: str
            The model task.
            One of [classification, regression]. 
        """

        super().__init__()
        valid_modes = ["classification", "regression"]
        assert mode in valid_modes, f"mode must be one of {valid_modes}"

        self.table = pd.read_csv(table_path)
        self.drug_dir = drug_dir
        self.cell_line_dir = cell_line_dir
        self.mode = mode

    def __len__(self):
        return self.table.shape[0]
    
    def __getitem__(self, idx):
        row = self.table.iloc[idx]
        drug_id = str(row["drug_id"])
        cell_line_id = row["cell_line_id"]
        target = row["label"] if self.mode == "classification" else row["ic50"]

        if self.mode == "classification":
            target = torch.tensor(target, dtype=torch.long)

        else:
            target = torch.tensor(target, dtype=torch.float32).unsqueeze(-1)
        
        drug_dir = os.path.join(self.drug_dir, drug_id)
        cell_line_dir = os.path.join(self.cell_line_dir, cell_line_id)

        drug_feature_path = os.path.join(drug_dir, "drug-feature.npy")
        drug_edge_list_path = os.path.join(drug_dir, "drug-edge-list.npy")

        gene_expression_path = os.path.join(cell_line_dir, "gene-feature.npy")
        methylation_path = os.path.join(cell_line_dir, "methylation-feature.npy")
        mutation_path = os.path.join(cell_line_dir, "mutation-feature.npy")

        gene_expression = np.load(gene_expression_path)
        methylation = np.load(methylation_path)
        mutation = np.load(mutation_path)

        drug_dict = {
            "feature_path": drug_feature_path,
            "edge_list_path": drug_edge_list_path
        }

        cell_line_dict = {
            "gene_expression": gene_expression,
            "methylation": methylation,
            "mutation": mutation
        }

        return drug_dict, cell_line_dict, target

def extract_graph(drug_dict: Dict[str, str]):

    """
    Extract graphs from a dictionary of graph paths.
    """

    features = [np.load(f) for f in drug_dict["feature_path"]]
    edge_lists = [np.load(f) for f in drug_dict["edge_list_path"]]

    features = [torch.tensor(f, dtype=torch.float32) for f in features]
    edge_lists = [torch.tensor(e, dtype=torch.long) for e in edge_lists]

    graphs = [GraphData(x=feature, edge_index=edge_list) for feature, edge_list in zip(features, edge_lists)]
    graph_loader = GraphDataLoader(graphs, batch_size=len(graphs), shuffle=False)
    graphs = next(iter(graph_loader))

    return graphs