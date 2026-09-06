import argparse
from pathlib import Path

import torch

import clip

def parse_args():
    parser = argparse.ArgumentParser(
        description="Encode the SUN-102 vocabulary with CLIP RN50."
    )
    parser.add_argument(
        "--vocabulary_path",
        default="experiment-results/exp01/sun102_vocabulary.txt",
    )
    parser.add_argument(
        "--output_path",
        default="data/exp01/sun102_clip_RN50_embeddings.pth",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Encode the SUN-102 vocabulary with CLIP RN50."
    )
    parser.add_argument(
        "--vocabulary_path",
        default="experiment-results/exp01/sun102_vocabulary.txt",
    )
    parser.add_argument(
        "--output_path",
        default="data/exp01/sun102_clip_RN50_embeddings.pth",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()

def load_concepts(vocabulary_path):
    with Path(vocabulary_path).open(encoding="utf-8") as file:
        concepts = [line.strip() for line in file if line.strip()]

    if len(concepts) != 102:
        raise ValueError(
            f"Expected 102 SUN concepts, found {len(concepts)}."
        )

    prompts = [f"a photo of {concept}" for concept in concepts]
    return concepts, prompts 

def main():
    args = parse_args()
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    concepts, prompts = load_concepts(args.vocabulary_path)

    model, _ = clip.load("RN50", device=args.device)
    model.eval()

    tokens = clip.tokenize(prompts).to(args.device)

    with torch.no_grad():
        embeddings = model.encode_text(tokens).float()
        embeddings = embeddings / embeddings.norm(
            dim=1, keepdim=True
        ).clamp_min(1e-12)

    payload = {
        "concepts": concepts,
        "prompts": prompts,
        "embeddings": embeddings.cpu(),
        "clip_model": "RN50",
        "prompt_template": "a photo of {concept}",
    }
    torch.save(payload, output_path)

    print(f"Concepts: {len(concepts)}")
    print(f"Embedding shape: {tuple(embeddings.shape)}")
    print(
        "Mean embedding norm:",
        embeddings.norm(dim=1).mean().item(),
    )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()