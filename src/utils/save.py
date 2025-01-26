from typing import Dict, Union

import numpy as np
import pandas as pd

def save_results(
    results: Dict[str, Union[str, np.ndarray]],
    reference_path: str,
    save_path: int
    ) -> None:

    """
    Saves the images, labels and embeddings from the dataset.

    Parameters
    ----------
    results: Dict[str, Union[str, np.ndarray]]
        A dictionary containing the images, labels and embeddings from the dataset.
    """

    df = pd.DataFrame(results)
    reference = pd.read_csv(reference_path)

    df["cell_line_id"] = reference["cell_line_id"]
    df["cancer_type"] = reference["cancer_type"]
    df["drug_id"] = reference["drug_id"]
    
    df.to_csv(save_path, index=False)