import argparse
import random
from pathlib import Path

from src.pipeline.inference import FacePipeline, _box_iou
from src.pipeline.preprocess_input import limit_size, load_image

ROOT = Path("data/raw/WIDER_FACE")
LABELS = ROOT / "wider_face_split/wider_face_val_bbx_gt.txt"
IMAGES = ROOT / "WIDER_val/images"
CONFIDENCE = 0.50
MATCH_IOU = 0.50
SEED = 42


def read_annotations(path):
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    index = 0

    while index < len(lines):
        name = lines[index]
        count = int(lines[index + 1])
        index += 2
        rows = lines[index:index + count]
        index += count

        # Một số mục không có mặt có thêm dòng số làm placeholder.
        if count == 0 and index < len(lines) and not lines[index].lower().endswith(".jpg"):
            index += 1

        valid, ignored = [], []
        for row in rows:
            fields = list(map(int, row.split()))
            x, y, width, height = fields[:4]
            if width <= 0 or height <= 0:
                continue
            box = (x, y, x + width, y + height)
            (ignored if fields[7] == 1 else valid).append(box)
        yield name, valid, ignored


def match_faces(faces, valid, ignored, scale):
    remaining = [tuple(value * scale for value in box) for box in valid]
    ignored = [tuple(value * scale for value in box) for box in ignored]
    tp = fp = 0

    for face in sorted(faces, key=lambda face: face.score, reverse=True):
        best = max(
            range(len(remaining)),
            key=lambda i: _box_iou(face.bbox, remaining[i]),
            default=None,
        )
        if best is not None and _box_iou(face.bbox, remaining[best]) >= MATCH_IOU:
            remaining.pop(best)
            tp += 1
        elif not any(_box_iou(face.bbox, box) >= MATCH_IOU for box in ignored):
            fp += 1

    return tp, fp, len(remaining)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit phải lớn hơn 0.")
    if not LABELS.is_file() or not IMAGES.is_dir():
        parser.error(f"Cần có file {LABELS} và thư mục {IMAGES}.")

    records = list(read_annotations(LABELS))
    random.Random(SEED).shuffle(records)
    records = records[:args.limit]

    pipeline = FacePipeline()
    pipeline.cfg["detection"]["nms_iou_threshold"] = 0.38
    detector = pipeline._get_detector()
    max_side = pipeline.cfg["preprocess"]["max_side"]
    tp = fp = fn = 0

    for i, (name, valid, ignored) in enumerate(records, start=1):
        image, _ = load_image(IMAGES / name)
        work, scale = limit_size(image, max_side)
        faces = pipeline._clean_detections(detector.detect(work, CONFIDENCE), work.shape)
        matched, extra, missed = match_faces(faces, valid, ignored, scale)
        tp += matched
        fp += extra
        fn += missed
        if i % 10 == 0 or i == len(records):
            print(f"Đã chấm {i}/{len(records)} ảnh", flush=True)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    print(f"Ảnh: {len(records)} | seed: {SEED} | confidence: {CONFIDENCE} | NMS: 0.38")
    print(f"IoU ghép: {MATCH_IOU} | TP: {tp} | FP: {fp} | FN: {fn}")
    print(f"Precision: {precision:.2%} | Recall: {recall:.2%}")


if __name__ == "__main__":
    main()