"""Create a small CLIP-RN50 feature tensor for EXP-00.

Run from the Discover-then-Name repository root. Images may be ordinary JPG,
JPEG, or PNG files placed in one folder; class subfolders are not required.
"""

import argparse
from pathlib import Path

import torch
from PIL import Image

from dncbm import arg_parser
from dncbm.utils import common_init, get_img_model


def parse_args():
    parser = arg_parser.get_common_parser()
    parser.set_defaults(img_enc_name="clip_RN50", device="cuda")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument(
        "--output_path",
        default="data/exp00/clip_RN50_pilot_features.pth",
    )
    return parser.parse_args()


def find_images(image_dir):
    extensions = {".jpg", ".jpeg", ".png"}
    paths = sorted(
        path for path in Path(image_dir).rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )
    if not paths:
        raise FileNotFoundError(f"No JPG, JPEG, or PNG images found in {image_dir}")
    return paths


def main():
    args = parse_args()
    common_init(args, disable_make_dirs=True)

    # Reuse the repository's CLIP loader and matching preprocessing transform.
    model, preprocess = get_img_model(args)
    model.eval()
    image_paths = find_images(args.image_dir)

    # A small batch is enough for EXP-00; no gradients or model updates are used.
    batch = torch.stack([
        preprocess(Image.open(path).convert("RGB")) for path in image_paths
    ]).to(args.device)

    with torch.no_grad():
        features = model.encode_image(batch).float().cpu()

    if features.ndim != 2 or features.shape[1] != 1024:
        raise RuntimeError(f"Expected [N, 1024] RN50 features, got {tuple(features.shape)}")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(features, output_path)
    print(f"Images: {len(image_paths)}")
    print(f"Feature shape: {tuple(features.shape)}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
