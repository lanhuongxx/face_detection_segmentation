"""Đánh giá RetinaFace trên WIDER FACE theo quy tắc của notebook."""
import argparse
import csv
import json
import random
import sys
from pathlib import Path
from time import perf_counter

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODELS = ("retinaface_mobile0.25", "retinaface_resnet50")
SIZE_BINS = ("<20", "20–<40", "40–<80", "≥80")
COUNT_KEYS = ("tp", "fp", "fn", "ignored")


def read_annotations(path):
    lines = iter(line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()
                 if line.strip())
    for name in lines:
        # Một số ảnh có 0 mặt đi kèm dòng toàn số 0.
        if name.split()[0].isdigit():
            if all(value == "0" for value in name.split()):
                continue
            raise ValueError(f"Dòng nhãn bất thường: {name}")
        count = int(next(lines))
        valid, ignored = [], []
        for _ in range(count):
            fields = list(map(int, next(lines).split()))
            x, y, width, height = fields[:4]
            if width > 0 and height > 0:
                boxes = ignored if fields[7] == 1 else valid
                boxes.append((x, y, x + width, y + height))
        yield name, valid, ignored



def box_iou(a, b):
    w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def match_faces(faces, valid, ignored, match_iou):
    matched = set()
    predictions = []
    for face in sorted(faces, key=lambda face: face.score, reverse=True):
        overlaps = [box_iou(face.bbox, box) for box in valid]
        eligible = [i for i, iou in enumerate(overlaps) if iou >= match_iou]
        available = [i for i in eligible if i not in matched]
        if available:
            matched.add(max(available, key=overlaps.__getitem__))
            status = "TP"
        elif not eligible and any(box_iou(face.bbox, box) >= match_iou for box in ignored):
            status = "ignored"
        else:
            status = "FP"
        predictions.append((face.bbox, status))
    missed = [box for i, box in enumerate(valid) if i not in matched]
    return predictions, missed, matched





def discover_dataset(root):
    labels = sorted(root.rglob("wider_face_val_bbx_gt.txt"))
    images = sorted(path for path in root.rglob("images")
                    if path.is_dir() and any(part.lower() == "wider_val"
                                            for part in path.relative_to(root).parts))
    if len(labels) != 1 or len(images) != 1:
        raise ValueError(f"{root}: cần đúng một file nhãn validation và thư mục WIDER_val/**/images.")
    return labels[0], images[0]


def select_records(labels, images, limit=200, seed=42, manifest=None):
    records = list(read_annotations(labels))
    if manifest is None:
        random.Random(seed).shuffle(records)
        selected = records[:limit]
    else:
        names = manifest.read_text(encoding="utf-8-sig").splitlines()
        names = [name.strip() for name in names if name.strip()]
        by_name = {row[0]: row for row in records}
        if len(names) != len(set(names)) or any(name not in by_name for name in names):
            raise ValueError("Danh sách ảnh chứa tên trùng hoặc tên không có trong nhãn.")
        selected = [by_name[name] for name in names]
    if not selected:
        raise ValueError("Cần ít nhất một ảnh.")
    missing = [name for name, _, _ in selected if not (images / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Thiếu {len(missing)} ảnh, ví dụ {missing[0]}")
    return selected


def size_group(box):
    side = min(box[2] - box[0], box[3] - box[1])
    for label, upper in zip(SIZE_BINS, (20, 40, 80)):
        if side < upper:
            return label
    return SIZE_BINS[-1]


def evaluate_image(record, pipeline, detector, images, confidence, match_iou):
    from src.pipeline.preprocess_input import limit_size, load_image

    name, valid, ignored = record
    image, _ = load_image(images / name)
    work, _ = limit_size(image, pipeline.cfg["preprocess"]["max_side"])
    sx, sy = work.shape[1] / image.shape[1], work.shape[0] / image.shape[0]

    def scaled(box):
        x1, y1, x2, y2 = box
        return x1 * sx, y1 * sy, x2 * sx, y2 * sy

    valid, ignored = list(map(scaled, valid)), list(map(scaled, ignored))
    start = perf_counter()
    faces = pipeline._clean_detections(detector.detect(work, confidence), work.shape)
    detection_ms = (perf_counter() - start) * 1000
    predictions, missed, matched = match_faces(faces, valid, ignored, match_iou)
    sizes = {label: {"tp": 0, "total": 0} for label in SIZE_BINS}
    for index, box in enumerate(valid):
        group = sizes[size_group(box)]
        group["total"] += 1
        group["tp"] += int(index in matched)
    return {
        "image": name, "tp": len(matched),
        "fp": sum(status == "FP" for _, status in predictions),
        "fn": len(missed),
        "ignored": sum(status == "ignored" for _, status in predictions),
        "detection_ms": detection_ms, "sizes": sizes,
        "predictions": predictions, "missed": missed,
    }


def summarize(rows):
    counts = {key: sum(row[key] for row in rows) for key in COUNT_KEYS}
    tp, fp, fn = (counts[key] for key in ("tp", "fp", "fn"))
    by_size = []
    for label in SIZE_BINS:
        hits = sum(row["sizes"][label]["tp"] for row in rows)
        total = sum(row["sizes"][label]["total"] for row in rows)
        by_size.append({"size_px": label, "tp": hits, "total": total,
                        "recall": hits / total if total else None})
    return {
        **counts, "precision": tp / (tp + fp) if tp + fp else 0,
        "recall": tp / (tp + fn) if tp + fn else 0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
        "recall_by_size": by_size,
        "mean_detection_ms": sum(row["detection_ms"] for row in rows[1:]) / (len(rows) - 1)
        if len(rows) > 1 else None,
    }


def evaluate(records, images, *, config, confidence=None, model=None, max_faces=None,
             nms_iou=0.38, match_iou=0.50):
    from src.pipeline.inference import FacePipeline
    from src.utils.config import load_config

    cfg = load_config(config)
    settings = cfg["detection"]
    overrides = {"model": model, "conf_threshold": confidence, "max_faces": max_faces,
                 "nms_iou_threshold": nms_iou}
    settings.update({key: value for key, value in overrides.items() if value is not None})
    confidence = settings["conf_threshold"]
    pipeline = FacePipeline(cfg=cfg)
    detector = pipeline._get_detector()
    rows = []
    for index, record in enumerate(records, 1):
        rows.append(evaluate_image(record, pipeline, detector, images, confidence, match_iou))
        if index % 50 == 0 or index == len(records):
            print(f"{settings['model']} · {index}/{len(records)} ảnh", flush=True)
    return {
        "dataset": "WIDER FACE validation", "detector": settings["model"],
        "confidence": confidence, "max_faces_per_image": settings["max_faces"],
        "nms_iou": settings["nms_iou_threshold"], "match_iou": match_iou,
        "max_side": cfg["preprocess"]["max_side"], "device": pipeline.device,
        "images_evaluated": len(rows), **summarize(rows),
    }, rows


def export_results(output, summary, rows):
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "selected_images.txt").write_text("".join(row["image"] + "\n" for row in rows), encoding="utf-8")
    for name, records, fields in [
        ("per_image.csv", rows, ["image", *COUNT_KEYS, "detection_ms"]),
        ("recall_by_size.csv", summary["recall_by_size"], ["size_px", "tp", "total", "recall"]),
    ]:
        with (output / name).open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw/WIDER_FACE"))
    parser.add_argument("--config", type=Path, default=Path("configs/pipeline.yaml"))
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--confidence", type=float)
    parser.add_argument("--max-faces", type=int)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--nms-iou", type=float, default=0.38)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--output", type=Path, default=Path("outputs/wider_face"))
    args = parser.parse_args()
    if args.limit < 1 or (args.max_faces is not None and args.max_faces < 1):
        parser.error("--limit và --max-faces phải lớn hơn 0.")
    if args.confidence is not None and not 0 <= args.confidence <= 1:
        parser.error("--confidence phải thuộc [0, 1].")
    if not 0 < args.nms_iou <= 1 or not 0 < args.match_iou <= 1:
        parser.error("Các ngưỡng IoU phải thuộc (0, 1].")
    if not args.config.is_file():
        parser.error(f"Không tìm thấy cấu hình: {args.config}")
    labels, images = discover_dataset(args.root)
    records = select_records(labels, images, args.limit, args.seed, args.manifest)
    summary, rows = evaluate(records, images, config=args.config, model=args.model,
                             confidence=args.confidence, max_faces=args.max_faces,
                             nms_iou=args.nms_iou, match_iou=args.match_iou)
    summary["seed"] = args.seed if args.manifest is None else None
    export_results(args.output, summary, rows)
    print(f"Precision={summary['precision']:.2%} | Recall={summary['recall']:.2%} | F1={summary['f1']:.2%}")
    print(f"Kết quả: {args.output.resolve()}")


if __name__ == "__main__":
    main()