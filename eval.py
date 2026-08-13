"""Evaluate the released ICML T-AVFD checkpoint on processed NPZ features."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import clip
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from model import FusionModel


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate T-AVFD on SHDF")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints" / "TAVFD.pt",
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=ROOT / "data" / "SHDF_features",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=ROOT / "metadata" / "SHDF_test.csv",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "SHDF_scores.csv",
    )
    return parser.parse_args()


def get_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    return device


def load_model(checkpoint_path: Path, device: torch.device) -> FusionModel:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    clip_model, _ = clip.load(
        checkpoint.get("clip_model", "ViT-L/14"), device=device, jit=False
    )
    model = FusionModel(
        clip_model,
        tm_weights=checkpoint.get("initial_tm_weights", [0.2, 0.2, -0.1]),
    ).to(device)
    missing, unexpected = model.load_state_dict(
        checkpoint["state_dict"], strict=False
    )
    contains_clip = any(
        key.startswith("text_feature.clip_model.")
        for key in checkpoint["state_dict"]
    )
    expected_missing = set()
    if not contains_clip:
        expected_missing = {
            key
            for key in model.state_dict()
            if key.startswith("text_feature.clip_model.")
        }
    if set(missing) != expected_missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}"
        )
    return model.eval()


def to_sequence(array: np.ndarray, name: str) -> torch.Tensor:
    tensor = torch.from_numpy(np.asarray(array)).float()
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D, got {tensor.shape}")
    return tensor


def load_npz(path: Path, device: torch.device) -> dict[str, torch.Tensor | None]:
    if not path.is_file():
        raise FileNotFoundError(f"Feature file not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        required = {"visual", "audio", "global"}
        if not required.issubset(data.files):
            raise KeyError(f"{path} needs NPZ keys {sorted(required)}")
        visual = to_sequence(data["visual"], "visual")
        audio = to_sequence(data["audio"], "audio")
        global_feature = to_sequence(data["global"], "global")
        local = to_sequence(data["local"], "local") if "local" in data else None

    if visual.shape != audio.shape or visual.shape[-1] != 1024:
        raise ValueError(f"Invalid audio/visual shapes in {path}")
    if global_feature.shape[-1] != 768:
        raise ValueError(f"Invalid global feature shape in {path}")
    return {
        "visual": F.normalize(visual, dim=-1).to(device),
        "audio": F.normalize(audio, dim=-1).to(device),
        "global": global_feature.to(device),
        "local": local.to(device) if local is not None else None,
    }


@torch.inference_mode()
def predict(model: FusionModel, path: Path, device: torch.device) -> float:
    features = load_npz(path, device)
    frame_logits = model(
        features["visual"],
        features["audio"],
        features["local"],
        features["global"],
    ).squeeze(0).squeeze(-1)
    return float(torch.logsumexp(-frame_logits, dim=0).cpu())


def read_metadata(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Metadata not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"path", "label"}.issubset(rows[0]):
        raise ValueError("Metadata must contain path and label columns")
    return rows


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    model = load_model(args.checkpoint, device)
    rows = read_metadata(args.metadata)

    results = []
    missing_files = []
    for row in tqdm(rows, desc="Evaluating SHDF", unit="video"):
        relative = Path(row["path"])
        path = relative if relative.is_absolute() else args.data_root / relative
        if not path.is_file():
            missing_files.append(str(path))
            tqdm.write(f"Skipping missing feature: {path}")
            continue
        results.append(
            {
                "path": row["path"],
                "label": int(row["label"]),
                "score": predict(model, path, device),
            }
        )

    if not results:
        raise RuntimeError("No NPZ files were found under the specified data root")

    labels = np.asarray([item["label"] for item in results])
    scores = np.asarray([item["score"] for item in results])
    if np.unique(labels).size != 2:
        raise ValueError("AP/AUC require both real and fake samples")
    metrics = {
        "num_videos": len(results),
        "num_missing": len(missing_files),
        "num_real": int((labels == 0).sum()),
        "num_fake": int((labels == 1).sum()),
        "ap": float(average_precision_score(labels, scores)),
        "auc": float(roc_auc_score(labels, scores)),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "score"])
        writer.writeheader()
        writer.writerows(results)
    args.output.with_suffix(".json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    if missing_files:
        print(f"Skipped {len(missing_files)} missing feature files")


if __name__ == "__main__":
    main()
