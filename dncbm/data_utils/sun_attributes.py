from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset


def load_matlab_strings(path, variable_name):
    """Convert a MATLAB cell array of strings into a Python list."""
    cells = loadmat(path)[variable_name].reshape(-1)
    strings = []

    for cell in cells:
        value = np.asarray(cell).squeeze()

        if value.size != 1:
            raise ValueError(
                f"Expected one string in each {variable_name} cell."
            )

        strings.append(str(value.item()))

    return strings