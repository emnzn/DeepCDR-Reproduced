from torch.utils.tensorboard import SummaryWriter

def log_metrics(
    writer: SummaryWriter, 
    mode: str, 
    loss: float,
    prefix: str,
    epoch: int,
    performance: float
    ):

    print(f"{prefix} Statistics:")
    writer.add_scalar(f"{prefix}/Loss", loss, epoch)

    if mode == "classification":
        print(f"Loss: {loss:.4f} | Balanced Accuracy: {performance:.4f}\n")
        writer.add_scalar(f"{prefix}/Balanced-Accuracy", performance, epoch)

    if mode == "regression":
        print(f"Loss: {loss:.4f} | Pearson Correlation: {performance:.4f}\n")
        writer.add_scalar(f"{prefix}/Pearson-Correlation", performance, epoch)
    