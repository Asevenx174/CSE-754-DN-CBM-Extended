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

class SUNAttributes(Dataset):
    """The 14,340-image annotated subset of SUNAttributes."""

    def __init__(
        self,
        annotation_dir,
        image_root,
        transform=None,
        majority_threshold=2 / 3,
    ):
        self.annotation_dir = Path(annotation_dir)
        self.image_root = Path(image_root)
        self.transform = transform

        self.attribute_names = load_matlab_strings(
            self.annotation_dir / "attributes.mat",
            "attributes",
        )

        self.relative_paths = load_matlab_strings(
            self.annotation_dir / "images.mat",
            "images",
        )

        labels = loadmat(
            self.annotation_dir / "attributeLabels_continuous.mat"
        )["labels_cv"]

        self.agreement = torch.from_numpy(labels).float()
        self.binary_labels = (
            self.agreement >= majority_threshold
        ).to(torch.uint8)

        expected_shape = (
            len(self.relative_paths),
            len(self.attribute_names),
        )

        if tuple(self.agreement.shape) != expected_shape:
            raise ValueError(
                f"Expected label shape {expected_shape}, "
                f"received {tuple(self.agreement.shape)}."
            )

        self.image_paths = [
            self.image_root / relative_path
            for relative_path in self.relative_paths
        ]

        missing = [
            path for path in self.image_paths
            if not path.is_file()
        ]

        if missing:
            raise FileNotFoundError(
                f"{len(missing)} annotated images are missing. "
                f"First missing image: {missing[0]}"
            )