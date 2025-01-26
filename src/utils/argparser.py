import os
import yaml
from typing import Dict, Union

def get_args(args_path: str) -> Dict[str, Union[float | str]]:
    """
    Gets relevant arguments from a yaml file.

    Parameters
    ----------
    args_path: str
        The path to the yaml file containing the arguments.
    
    Returns
    -------    
    args: Dict[str, Union[float, str]]
        The arguments in the form of a dictionary.
    """

    with open(args_path, "r") as f:
        args = yaml.safe_load(f)

    return args


def save_args(
    args: Dict[str, Union[float, str]], 
    dest_dir: str
    ):
    """
    Saves the arguments as a yaml file in a given destination directory.
    
    Parameters
    ----------
    args: Dict[str, Union[float, str]]
        The arguments in the form of a dictionary.

    dest_dir: str
        The the directory of the save destination.
    """

    path = os.path.join(dest_dir, "run-config.yaml")
    with open(path, "w") as f:
        yaml.dump(args, f)