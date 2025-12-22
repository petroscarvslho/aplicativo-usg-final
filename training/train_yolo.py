#!/usr/bin/env python3
"""
Treino YOLO para plugins de deteccao (NEEDLE/FAST).
Gera dataset YOLO a partir dos exports .npy do unified_dataset_manager.
"""

import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2

from plugin_registry import get_model_spec


BASE_DIR = Path(__file__).parent
DATASETS_DIR = BASE_DIR.parent / "datasets" / "unified" / "exports"
YOLO_EXPORTS_DIR = BASE_DIR.parent / "datasets" / "unified" / "exports_yolo"


def write_yolo_dataset(images: np.ndarray, labels: np.ndarray, split_dir: Path, class_count: int):
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for idx, (img, label) in enumerate(zip(images, labels)):
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img.squeeze(-1)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        img_name = f"{idx:06d}.png"
        cv2.imwrite(str(images_dir / img_name), img)

        label_path = labels_dir / f"{idx:06d}.txt"
        if label.ndim == 1:
            entries = [label]
        else:
            entries = label

        lines = []
        for entry in entries:
            if len(entry) == 2:
                y, x = entry
                cls = 0
                w = 0.06
                h = 0.06
            elif len(entry) == 5:
                cls, x, y, w, h = entry
                cls = int(cls)
            else:
                continue

            if cls < 0 or cls >= class_count:
                continue
            lines.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

        label_path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description="YOLO Training (NEEDLE/FAST)")
    parser.add_argument("--plugin", required=True, help="Plugin (NEEDLE/FAST)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--base_model", type=str, default="yolov8n.pt")
    parser.add_argument("--data_dir", type=str, default=None)
    args = parser.parse_args()

    spec = get_model_spec(args.plugin, "yolo")
    if not spec:
        print(f"ERROR: Plugin/Modelo YOLO desconhecido: {args.plugin}")
        return 1

    data_dir = Path(args.data_dir) if args.data_dir else DATASETS_DIR / args.plugin.lower()
    x_train = data_dir / "X_train.npy"
    y_train = data_dir / "Y_train.npy"
    x_val = data_dir / "X_val.npy"
    y_val = data_dir / "Y_val.npy"

    if not x_train.exists():
        print(f"ERROR: Dataset nao encontrado em {data_dir}")
        print("Execute: python datasets/unified_dataset_manager.py")
        return 1

    images_train = np.load(x_train)
    labels_train = np.load(y_train)
    images_val = np.load(x_val)
    labels_val = np.load(y_val)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_dir = YOLO_EXPORTS_DIR / args.plugin.lower() / timestamp
    train_dir = export_dir / "train"
    val_dir = export_dir / "val"
    export_dir.mkdir(parents=True, exist_ok=True)

    class_names = spec.get("class_names", ["object"])
    write_yolo_dataset(images_train, labels_train, train_dir, len(class_names))
    write_yolo_dataset(images_val, labels_val, val_dir, len(class_names))

    data_yaml = export_dir / "data.yaml"
    data_yaml.write_text(
        "\n".join([
            f"path: {export_dir}",
            "train: train/images",
            "val: val/images",
            f"nc: {len(class_names)}",
            f"names: {class_names}",
        ])
    )

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics nao instalado. Rode: pip install ultralytics")
        return 1

    model = YOLO(args.base_model)
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch_size,
        project=str(export_dir),
        name="train",
    )

    best_path = export_dir / "train" / "weights" / "best.pt"
    if best_path.exists():
        target_path = Path(spec["expected_path"])
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_path, target_path)
        meta = {
            "plugin": args.plugin.upper(),
            "model": "yolo",
            "source": str(best_path),
            "exported_at": datetime.now().isoformat(),
        }
        meta_path = target_path.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"OK: pesos exportados para {target_path}")
    else:
        print("WARN: best.pt nao encontrado apos o treino.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
