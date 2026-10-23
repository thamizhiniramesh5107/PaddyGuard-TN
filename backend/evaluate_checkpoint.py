"""Evaluate an already trained checkpoint on a deterministic held-out split."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v3_small
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from torch import nn


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.digest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="paddy-disease-classification")
    parser.add_argument("--checkpoint", default="artifacts/paddyguard.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=24)
    args = parser.parse_args()
    root = Path(args.data)
    image_root = root / "train_images" if (root / "train_images").is_dir() else root
    ds = datasets.ImageFolder(image_root)
    seen, keep = set(), []
    for i, (filename, _) in enumerate(ds.samples):
        sig = digest(filename)
        if sig not in seen:
            seen.add(sig)
            keep.append(i)
    labels = np.asarray(ds.targets)[keep]
    pool, test_local = train_test_split(np.arange(len(keep)), test_size=.15,
                                        random_state=args.seed, stratify=labels)
    train_local, val_local = train_test_split(pool, test_size=.15 / .85,
                                              random_state=args.seed, stratify=labels[pool])
    test_idx = np.asarray(keep)[test_local]
    checkpoint = Path(args.checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    classes = ds.classes
    if state.get("classes") != classes:
        raise ValueError("Checkpoint class names/order do not match the labeled dataset")
    model = mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(classes))
    model.load_state_dict(state["model"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    eval_ds = datasets.ImageFolder(image_root, transform=transform)
    loader = DataLoader(Subset(eval_ds, test_idx), batch_size=args.batch_size,
                        shuffle=False, num_workers=0)
    predicted, actual = [], []
    with torch.inference_mode():
        for images, targets in loader:
            predicted.extend(model(images.to(device)).argmax(1).cpu().tolist())
            actual.extend(targets.tolist())
    p, r, f, _ = precision_recall_fscore_support(
        actual, predicted, labels=list(range(len(classes))), zero_division=0)
    metrics = {
        "architecture": "MobileNetV3-Small",
        "evaluation_split": "deterministic stratified held-out 15% of labeled training data; unlabeled competition test_images excluded",
        "test_samples": len(actual),
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision_macro": float(p.mean()), "recall_macro": float(r.mean()),
        "f1_macro": float(f.mean()),
        "per_class": classification_report(actual, predicted, labels=list(range(len(classes))),
                                           target_names=classes, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=list(range(len(classes)))).tolist(),
        "classes": classes,
        "split_counts": {"train": len(train_local), "validation": len(val_local), "test": len(test_idx)},
        "duplicates_removed": len(ds.samples) - len(keep), "seed": args.seed,
        "checkpoint": str(checkpoint),
        "best_epoch": state.get("best_epoch"),
        "best_validation_accuracy": state.get("best_validation_accuracy"),
    }
    out = checkpoint.parent / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_class"}, indent=2))


if __name__ == "__main__":
    main()
