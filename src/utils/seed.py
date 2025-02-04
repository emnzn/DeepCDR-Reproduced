import torch
import random
import numpy as np

def set_seed(seed: int) -> None:
    """
    Sets the seed for reproducibility.

    Parameters
    ----------
    seed: int
        The seed to use.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True