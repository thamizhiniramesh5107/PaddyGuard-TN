"""Train a MobileNetV3-Small paddy classifier on labeled ImageFolder data."""
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/images", help="Labeled image root, class-folder root, or Kaggle dataset folder")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=int, default=6)
    parser.add_argument("--resume", action="store_true", help="Resume fine-tuning from artifacts/paddyguard.pt for the requested number of additional epochs")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("--epochs and --batch-size must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, args.cpu_threads))
    root = Path(args.data)
    if not root.is_dir():
        raise FileNotFoundError(f"Image dataset folder not found: {root}. Provide labeled image class folders; an unlabeled competition test folder cannot train or evaluate a model.")
    # This Kaggle dataset stores CSV labels and images in matching class subfolders.
    image_root = root / "train_images" if (root / "train_images").is_dir() else root
    train_root = next((image_root / name for name in ("train", "training") if (image_root / name).is_dir()), image_root)
    val_root = next((image_root / name for name in ("val", "valid", "validation") if (image_root / name).is_dir()), None)
    test_root = next((image_root / name for name in ("test", "testing") if (image_root / name).is_dir()), None)
    train_eval = datasets.ImageFolder(train_root)
    classes = train_eval.classes
    if len(classes) < 2:
        raise ValueError(f"Expected labeled class subfolders under {train_root}; found {len(classes)} class(es)")
    if not train_eval.samples:
        raise ValueError(f"No supported image files found under {train_root}")
    # Honor the supplied CSV-to-folder labels when this is the Paddy Disease Classification dataset.
    csv_path = root / "train.csv" if root != train_root else root.parent / "train.csv"
    if csv_path.exists():
        with csv_path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        csv_labels = {row.get("label", "") for row in rows}
        if csv_labels and csv_labels != set(classes):
            raise ValueError(f"train.csv labels do not match image folders: CSV={sorted(csv_labels)}, folders={classes}")
        missing_rows = [row.get("image_id", "") for row in rows
                        if row.get("label") in classes and not (train_root / row["label"] / row["image_id"]).is_file()]
        if missing_rows:
            raise ValueError(f"{len(missing_rows)} images listed in {csv_path.name} are missing; first: {missing_rows[0]}")
    # Deduplicate byte-identical images before splitting to prevent train/test leakage.
    seen = {}
    keep = []
    duplicate_count = 0
    for index, (filename, label_index) in enumerate(train_eval.samples):
        digest = sha256_file(filename)
        if digest in seen:
            prior_label, prior_file = seen[digest]
            if prior_label != label_index:
                raise ValueError(f"Identical image has conflicting labels: {prior_file} and {filename}")
            duplicate_count += 1
            continue
        seen[digest] = (label_index, filename)
        keep.append(index)
    labels = np.asarray([train_eval.targets[i] for i in keep])
    print(f"Dataset: {len(train_eval.samples)} images, {len(keep)} unique, {duplicate_count} exact duplicates removed; classes={classes}")
    if val_root is None and test_root is None:
        pool_idx, test_local = train_test_split(np.arange(len(keep)), test_size=.15, random_state=args.seed, stratify=labels)
        train_local, val_local = train_test_split(pool_idx, test_size=.15 / .85, random_state=args.seed, stratify=labels[pool_idx])
        train_idx = np.asarray(keep)[train_local]
        val_idx = np.asarray(keep)[val_local]
        test_idx = np.asarray(keep)[test_local]
        val_eval = train_eval
        test_eval = train_eval
    else:
        train_idx = np.asarray(keep)
        if val_root is None:
            train_idx, val_local = train_test_split(train_idx, test_size=.15, random_state=args.seed, stratify=np.asarray(train_eval.targets)[train_idx])
            val_idx = val_local
            val_eval = train_eval
        else:
            val_eval = datasets.ImageFolder(val_root)
            if val_eval.classes != classes:
                raise ValueError("Validation folder classes must exactly match training folder classes")
            val_idx = np.arange(len(val_eval))
        if test_root is None:
            train_idx, test_local = train_test_split(train_idx, test_size=.15, random_state=args.seed, stratify=np.asarray(train_eval.targets)[train_idx])
            test_idx = test_local
            test_eval = train_eval
        else:
            test_eval = datasets.ImageFolder(test_root)
            if test_eval.classes != classes:
                raise ValueError("Test folder classes must exactly match training folder classes")
            test_idx = np.arange(len(test_eval))
    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError("Train, validation and test partitions must all contain images")
    train_targets = np.asarray(train_eval.targets)
    counts = np.bincount(train_targets[train_idx], minlength=len(classes))
    if np.any(counts == 0):
        raise ValueError("Every disease class must have at least one training image")
    train_tf = transforms.Compose([
        transforms.Resize((232, 232)),
        transforms.RandomResizedCrop(224, scale=(.82, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=.15, contrast=.15, saturation=.12, hue=.025),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    train_ds = datasets.ImageFolder(train_root, transform=train_tf)
    val_ds = datasets.ImageFolder(val_root, transform=eval_tf) if val_root else datasets.ImageFolder(train_root, transform=eval_tf)
    test_ds = datasets.ImageFolder(test_root, transform=eval_tf) if test_root else datasets.ImageFolder(train_root, transform=eval_tf)
    sample_weights = torch.as_tensor((1.0 / counts)[train_targets[train_idx]], dtype=torch.double)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_idx), replacement=True)
    train_dl = DataLoader(Subset(train_ds, train_idx), batch_size=args.batch_size, sampler=sampler, num_workers=0)
    val_dl = DataLoader(Subset(val_ds, val_idx), batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_dl = DataLoader(Subset(test_ds, test_idx), batch_size=args.batch_size, shuffle=False, num_workers=0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(__file__).resolve().parent.parent / "artifacts"
    out.mkdir(exist_ok=True)
    checkpoint = out / "paddyguard.pt"
    if args.resume:
        if not checkpoint.exists():
            raise FileNotFoundError(f"Cannot resume: checkpoint does not exist: {checkpoint}")
        prior = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if prior.get("classes") != classes or prior.get("architecture") != "mobilenet_v3_small":
            raise ValueError("Resume checkpoint architecture/classes do not match this dataset")
        model = mobilenet_v3_small(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(classes))
        model.load_state_dict(prior["model"])
        best_score = float(prior.get("best_validation_accuracy", -1.0))
        best_epoch = int(prior.get("best_epoch", 0))
        for param in model.features[-3:].parameters():
            param.requires_grad = True
        optimizer = torch.optim.AdamW([
            {"params": model.features[-3:].parameters(), "lr": 1e-4},
            {"params": model.classifier.parameters(), "lr": 5e-4},
        ])
        print(f"Resuming from epoch {best_epoch}, validation_accuracy={best_score:.4f}", flush=True)
    else:
        try:
            model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        except Exception as exc:
            raise RuntimeError("Could not load pretrained MobileNetV3-Small weights. Connect once to download the official torchvision weights, then rerun.") from exc
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(classes))
        for param in model.features.parameters():
            param.requires_grad = False
        best_score = -1.0
        best_epoch = 0
        optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        if not args.resume and epoch == 2 and args.epochs > 2:
            for param in model.features[-3:].parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW([
                {"params": model.features[-3:].parameters(), "lr": 1e-4},
                {"params": model.classifier.parameters(), "lr": 5e-4},
            ])
        model.train()
        total_loss = 0.0
        for images, targets in train_dl:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        model.eval()
        val_pred, val_true = [], []
        with torch.inference_mode():
            for images, targets in val_dl:
                val_pred.extend(model(images.to(device)).argmax(1).cpu().numpy())
                val_true.extend(targets.numpy())
        val_score = accuracy_score(val_true, val_pred)
        print(f"Epoch {epoch + 1}/{args.epochs}: loss={total_loss / max(1, len(train_dl)):.4f}, validation_accuracy={val_score:.4f}", flush=True)
        if val_score > best_score:
            best_score, best_epoch = val_score, epoch + 1
            torch.save({"model": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "classes": classes, "architecture": "mobilenet_v3_small", "input_size": 224, "best_epoch": best_epoch, "best_validation_accuracy": best_score}, checkpoint)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    model.eval()
    test_pred, test_true = [], []
    with torch.inference_mode():
        for images, targets in test_dl:
            test_pred.extend(model(images.to(device)).argmax(1).cpu().numpy())
            test_true.extend(targets.numpy())
    precision, recall, f1, _ = precision_recall_fscore_support(test_true, test_pred, labels=list(range(len(classes))), zero_division=0)
    metrics = {
        "architecture": "MobileNetV3-Small",
        "evaluation_split": "held-out labeled test partition; unlabeled competition test_images were not used",
        "test_samples": len(test_true),
        "accuracy": float(accuracy_score(test_true, test_pred)),
        "precision_macro": float(precision.mean()),
        "recall_macro": float(recall.mean()),
        "f1_macro": float(f1.mean()),
        "per_class": classification_report(test_true, test_pred, labels=list(range(len(classes))), target_names=classes, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(test_true, test_pred, labels=list(range(len(classes)))).tolist(),
        "classes": classes,
        "split_counts": {"train": len(train_idx), "validation": len(val_idx), "test": len(test_idx)},
        "duplicates_removed": duplicate_count,
        "seed": args.seed,
        "best_epoch": best_epoch,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in metrics.items() if key != "per_class"}, indent=2), flush=True)

if __name__ == "__main__":
    main()
