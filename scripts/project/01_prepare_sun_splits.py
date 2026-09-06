import argparse
import csv
import json
from pathlib import Path

import torch

from dncbm.data_utils.sun_attributes import SUNAttributes


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--annotation_dir",
        default=(
            "data/sunattributes/"
            "SUNAttributeDB/SUNAttributeDB"
        ),
    )

    parser.add_argument(
        "--image_root",
        default=(
            "data/sunattributes/"
            "SUNAttributeDB_Images/images"
        ),
    )

    parser.add_argument(
        "--output_dir",
        default="experiment-results/exp01",
    )

    parser.add_argument("--seed", type=int, default=49)
    parser.add_argument("--minimum_test_support", type=int, default=30)

    return parser.parse_args()