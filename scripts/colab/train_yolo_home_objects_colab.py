#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Colab-friendly YOLO training launcher for URIS home-object detection."
    )
    parser.add_argument("--data", required=True, help="YOLO dataset YAML path (train/val names + class map)")
    parser.add_argument("--model", default="yolov8s.pt", help="Base model checkpoint (e.g., yolov8n.pt / yolov8s.pt)")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="./runs/uris-yolo")
    parser.add_argument("--name", default="home-objects-v1")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--conf", type=float, default=0.25, help="Validation confidence threshold")
    parser.add_argument("--save-json", action="store_true", help="Save COCO-style JSON metrics when supported")
    parser.add_argument("--export", default="onnx", choices=["none", "onnx", "torchscript", "engine"])
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover - env specific
        print("ultralytics is not installed. In Colab run: !pip install ultralytics")
        print(f"Import error: {exc}")
        return 1

    data_path = str(Path(args.data).expanduser())
    print("URIS YOLO Colab Training")
    print(json.dumps(vars(args), indent=2, ensure_ascii=False))

    model = YOLO(args.model)
    train_results = model.train(
        data=data_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        pretrained=True,
        cache=True,
        plots=True,
        verbose=True,
    )
    print("Train run complete")
    print(train_results)

    val_results = model.val(data=data_path, conf=args.conf, save_json=args.save_json)
    print("Validation complete")
    print(val_results)

    if args.export != "none":
        try:
            export_results = model.export(format=args.export)
            print(f"Export complete: {export_results}")
        except Exception as exc:  # pragma: no cover - export backend availability varies
            print(f"Export skipped/failed: {exc}")

    print("Next step:")
    print("- Use the trained detector in URIS Live Camera path to improve object_registry stability and reference resolution.")
    print("- Evaluate on your hard split (遮挡/低照/多同类物体).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
