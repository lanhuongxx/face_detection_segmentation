"""Tune Haar trên 500 ảnh WIDER train cố định rồi khóa cấu hình.

Thử lưới tham số, mỗi cấu hình chọn ngưỡng score cho F1 cao nhất, lấy cấu hình
có F1 cao nhất. Kết quả ghi vào configs/detection_locked.yaml; phải làm bước này
TRƯỚC khi chạy WIDER val và không sửa tay sau đó.
"""

from __future__ import annotations

import argparse
import itertools
import json

import cv2
import yaml

from src.detection.detect_metrics import evaluate_detections
from src.detection.haar_baseline import HaarFaceDetector
from src.detection.run_detect import _DEFAULT_CONFIG, load_config, load_manifest
from src.detection.wider_data import load_ground_truth

# Lưới tham số thử: 3 x 4 x 2 x 2 = 48 cấu hình
GRID = {
    "scale_factor": [1.05, 1.1, 1.2],
    "min_neighbors": [2, 3, 5, 8],
    "min_size": [[24, 24], [30, 30]],
    "equalize_hist": [False, True],
}


def tune(images, gt, grid=GRID):
    """images: {id: ảnh xám}. Trả danh sách kết quả, mỗi cấu hình một dòng."""
    results = []
    # itertools.product thử hết mọi tổ hợp trong lưới
    for sf, mn, ms, eq in itertools.product(*grid.values()):
        det = HaarFaceDetector(scale_factor=sf, min_neighbors=mn, min_size=tuple(ms), equalize_hist=eq)
        preds = {i: [{"bbox": f["bbox"], "score": f["score"]} for f in det.detect(img)]
                 for i, img in images.items()}
        # gt là nhãn của cả 12880 ảnh train nhưng ở đây chỉ chạy 500 ảnh,
        # nên phải để require_full=False, không thì báo "Thiếu prediction" và dừng
        r = evaluate_detections(preds, gt, size_groups=None, require_full=False)["all"]
        # nếu cấu hình không ra box nào thì không có best_f1, cho F1 = 0
        best = r.get("best_f1", {"score_thr": 1.0, "f1": 0.0, "precision": 0.0, "recall": 0.0})
        row = {"scale_factor": sf, "min_neighbors": mn, "min_size": list(ms), "equalize_hist": eq, **best}
        print(f"sf={sf} mn={mn} ms={ms} eq={eq}: F1={row['f1']:.4f} thr={row['score_thr']:.3f}")
        results.append(row)
    return results


def main():
    p = argparse.ArgumentParser(description="Tune Haar trên tập 500 ảnh train.")
    p.add_argument("--manifest", required=True, help="manifest_wider_tune500.csv")
    p.add_argument("--gt", required=True, help="wider_face_train_bbx_gt.txt")
    p.add_argument("--config", default=str(_DEFAULT_CONFIG))
    p.add_argument("--locked-config", default="configs/detection_locked.yaml")
    p.add_argument("--out", default="outputs/detection/haar_tuning.json")
    a = p.parse_args()

    rows = load_manifest(a.manifest, split="tune")
    # Haar chỉ cần ảnh xám nên đọc 1 lần, dùng lại cho mọi cấu hình
    images = {i: cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) for i, path in rows}
    results = tune(images, load_ground_truth(a.gt))
    # chọn cấu hình có F1 cao nhất
    best = max(results, key=lambda r: r["f1"])
    print("Tốt nhất:", best)

    # lấy config gốc rồi ghi đè phần haar bằng cấu hình tốt nhất vừa tìm được
    config = load_config(a.config)
    config["haar"] = {k: best[k] for k in ("scale_factor", "min_neighbors", "min_size", "equalize_hist")}
    config["haar"]["score_thr"] = round(best["score_thr"], 6)
    config.setdefault("retinaface", {})["score_thr"] = 0.5  # RetinaFace cố định 0.5
    with open(a.locked_config, "w", encoding="utf-8") as f:
        f.write("# Cấu hình đã khóa bởi tune_haar.py, không sửa tay sau khi chạy WIDER val\n")
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
    import os
    # tạo thư mục outputs/detection nếu chưa có rồi ghi toàn bộ kết quả tune, xếp F1 cao xuống thấp
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(sorted(results, key=lambda r: -r["f1"]), f, indent=2)
    print("Đã ghi", a.locked_config)


if __name__ == "__main__":
    main()