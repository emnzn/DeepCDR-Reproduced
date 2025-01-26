import os
from math import inf
from typing import Tuple
from datetime import datetime

import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import balanced_accuracy_score


from utils import (
    get_args,
    extract_graph,
    log_metrics,
    DeepCDR,
    MultiOmicsDataset
    )

def train(
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer, 
    model: DeepCDR,
    mode: str,
    device: str
    ) -> Tuple[float, float]:

    """
    Trains the model for one epoch.

    Parameters
    ----------
    dataloader: DataLoader
        The data loader to iterate over.

    criterion: nn.Module
        The loss function.

    optimizer: optim.Optimizer
        The optimizer for parameter updates.

    model: Network
        The model to be trained.

    mode: str
        The model task.
        One of [classification, regression].

    device: str
        One of [cuda, cpu].

    Returns
    -------
    epoch_loss: float
        The average loss for the given epoch.

    epoch_balanced_accuracy: float
        The average balanced accuracy for the given epoch.  
    """

    metrics = {
        "running_loss": 0,
        "predictions": [],
        "targets": []
    }
    
    model.train()
    for drug_dict, cell_line_dict, target in tqdm(dataloader, desc="Training in progress"):
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

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if mode == "classification":
            confidence = F.softmax(out, dim=1)
            pred = torch.argmax(confidence, dim=1)

        if mode == "regression":
            pred = out

        metrics["running_loss"] += loss.detach().cpu().item()
        metrics["predictions"].extend(pred.detach().cpu().numpy().squeeze(-1))
        metrics["targets"].extend(target.cpu().numpy().squeeze(-1))

    epoch_loss = metrics["running_loss"] / len(dataloader)

    if mode == "classification":
        performance = balanced_accuracy_score(metrics["targets"], metrics["predictions"])

    if mode == "regression":
        performance, _ = pearsonr(metrics["targets"], metrics["predictions"])

    return epoch_loss, performance

@torch.no_grad()
def validate(
    dataloader: DataLoader,
    criterion: nn.Module,
    model: DeepCDR,
    mode: str,
    device: str
    ):

    """
    Runs validation for a single epoch.
    """
    
    metrics = {
        "running_loss": 0,
        "predictions": [],
        "targets": []
    }

    model.eval()
    for drug_dict, cell_line_dict, target in tqdm(dataloader, desc="Validation in progress"):
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

        if mode == "regression":
            pred = out

        metrics["running_loss"] += loss.cpu().item()
        metrics["predictions"].extend(pred.cpu().numpy().squeeze(-1))
        metrics["targets"].extend(target.cpu().numpy().squeeze(-1))

    epoch_loss = metrics["running_loss"] / len(dataloader)
    
    if mode == "classification":
        performance = balanced_accuracy_score(metrics["targets"], metrics["predictions"])

    if mode == "regression":
        performance, _ = pearsonr(metrics["targets"], metrics["predictions"])

    return epoch_loss, performance


def main():
    arg_path = os.path.join("configs", "train.yaml")
    args = get_args(arg_path)
    identifier = datetime.now().strftime("%m-%d-%Y_%H-hrs") 

    model_dir = os.path.join("..", "assets", "models", args["mode"], identifier)
    os.makedirs(model_dir, exist_ok=True)

    data_dir = os.path.join("..", "data", "cleaned")
    log_dir = os.path.join("runs", args["mode"], identifier)

    writer = SummaryWriter(log_dir)

    train_dataset = MultiOmicsDataset(
        table_path=os.path.join(data_dir, "train.csv"),
        drug_dir=os.path.join(data_dir, "drugs"),
        cell_line_dir=os.path.join(data_dir, "cell-line"),
        mode=args["mode"]
    )

    val_dataset = MultiOmicsDataset(
        table_path=os.path.join(data_dir, "validation.csv"),
        drug_dir=os.path.join(data_dir, "drugs"),
        cell_line_dir=os.path.join(data_dir, "cell-line"),
        mode=args["mode"]
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("cuda available: ", torch.cuda.is_available())

    train_loader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args["batch_size"], shuffle=False)

    model = DeepCDR(
        mode=args["mode"],
        output_dim=args["output_dim"],
        dropout_prob=args["dropout_prob"]
    ).to(device)

    criterion = nn.CrossEntropyLoss() if args["mode"] == "classification" else nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args["learning_rate"], weight_decay=args["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args["epochs"], eta_min=args["eta_min"])

    min_val_loss, max_val_performance = inf, -inf

    for epoch in range(1, args["epochs"] + 1):
        print(f"Epoch [{epoch}/{args['epochs']}]")
        
        train_loss, train_performance = train(
            dataloader=train_loader, 
            criterion=criterion, 
            optimizer=optimizer, 
            model=model, 
            mode=args["mode"],
            device=device
            )

        log_metrics(
            writer=writer,
            mode=args["mode"],
            loss=train_loss,
            prefix="Train",
            epoch=epoch,
            performance=train_performance
        )

        val_loss, val_performance = validate(
            dataloader=val_loader, 
            criterion=criterion, 
            model=model, 
            mode=args["mode"],
            device=device
            )

        log_metrics(
            writer=writer,
            mode=args["mode"],
            loss=val_loss,
            prefix="Validation",
            epoch=epoch,
            performance=val_performance
        )

        if val_loss < min_val_loss:
            torch.save(model.state_dict(), os.path.join(model_dir, f"lowest-loss.pth"))
            min_val_loss = val_loss
            print("New minimum loss — model saved.")
        
        if val_performance > max_val_performance:
            save_filename = "highest-balanced-accuracy.pth" if args["mode"] == "classification" else "highest-pearson-correlation.pth"
            torch.save(model.state_dict(), os.path.join(model_dir, save_filename))
            max_val_performance = val_performance
            
            print(f"New maximum {' '.join(save_filename.split('-')[1:2])} — model saved.")
    
        scheduler.step()

        print("-------------------------------------------------------------------\n")

    performance_metric = "Max Balanced Accuracy" if args["mode"] == "classification" else "Max Pearson Correlation"
    print("Run Summary:")
    print(f"Min Loss: {min_val_loss:.4f} | {performance_metric}: {max_val_performance:.4f}\n")

        
if __name__ == "__main__":
    main()  