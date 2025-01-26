import os
from typing import Tuple

import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score

from utils import (
    get_args,
    save_results,
    extract_graph,
    DeepCDR,
    MultiOmicsDataset
)

@torch.no_grad()
def inference(
    dataloader: DataLoader,
    criterion: nn.Module,
    model: DeepCDR,
    mode: str,
    device: str,
    reference_path: str,
    save_dir: str
    ) -> Tuple[float, float]:

    """
    Runs inference.
    """
    
    metrics = {
        "loss": [],
        "prediction": [],
        "target": [],
    }

    model.eval()
    for drug_dict, cell_line_dict, target in tqdm(dataloader, desc="Inference in progress"):
        drug_graphs = extract_graph(drug_dict).to(device)
        gene_expression = cell_line_dict["gene_expression"].to(device)
        methylation = cell_line_dict["methylation"].to(device)
        mutation = cell_line_dict["mutation"].to(device)
        target = target.to(device)

        out = model(
            drug_graphs, 
            gene_expression,
            methylation,
            mutation
        )
        loss = criterion(out, target)

        if mode == "classification":
            confidence = F.softmax(out, dim=1)
            pred = torch.argmax(confidence, dim=1)

            metrics["prediction"].extend(pred.cpu().numpy())
            metrics["target"].extend(target.cpu().numpy())

        if mode == "regression":
            pred = out

            metrics["prediction"].extend(pred.cpu().numpy().squeeze(-1))
            metrics["target"].extend(target.cpu().numpy().squeeze(-1))

        metrics["loss"].extend(loss.squeeze().cpu().numpy())

    epoch_loss = sum(metrics["loss"]) / len(dataloader)
    
    if mode == "classification":
        performance = balanced_accuracy_score(metrics["target"], metrics["prediction"])

    if mode == "regression":
        performance, _ = pearsonr(metrics["target"], metrics["prediction"])

    save_results(metrics, reference_path, os.path.join(save_dir, "results.csv"))

    return epoch_loss, performance


def main():
    arg_path = os.path.join("configs", "inference.yaml")
    args = get_args(arg_path)
    identifier = args["identifier"]

    data_dir = os.path.join("..", "data", "cleaned")
    save_dir = os.path.join("..", "assets", "inference-tables", args["mode"], identifier)
    os.makedirs(save_dir, exist_ok=True)

    run_path = os.path.join("runs", args["mode"], identifier, "run-config.yaml")
    weight_path = os.path.join("..", "assets", "models", args["mode"], identifier, args["weights"])
    run_args = get_args(run_path)

    inference_dataset = MultiOmicsDataset(
        table_path=os.path.join(data_dir, "test.csv"),
        drug_dir=os.path.join(data_dir, "drugs"),
        cell_line_dir=os.path.join(data_dir, "cell-line"),
        mode=args["mode"]
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("cuda available: ", torch.cuda.is_available())

    inference_loader = DataLoader(inference_dataset, batch_size=args["batch_size"], shuffle=False)

    model = DeepCDR(
        mode=args["mode"],
        output_dim=run_args["output_dim"],
        dropout_prob=run_args["dropout_prob"]
    ).to(device)

    weights = torch.load(weight_path, map_location=torch.device(device), weights_only=True)
    model.load_state_dict(weights)

    if args["mode"] == "classification":
        criterion = nn.CrossEntropyLoss(reduction="none")
        performance_metric = "Balanced Accuracy"

    if args["mode"] == "regression": 
        criterion = nn.MSELoss(reduction="none")
        performance_metric = "Pearson Correlation"

    loss, performance = inference(
        dataloader=inference_loader,
        criterion=criterion,
        model=model,
        mode=args["mode"],
        device=device,
        reference_path=os.path.join(data_dir, "test.csv"),
        save_dir=save_dir
    )

    print("\nInference Summary:")
    print(f"Loss: {loss:.4f} | {performance_metric}: {performance:.4f}\n")


if __name__ == "__main__":
    main()