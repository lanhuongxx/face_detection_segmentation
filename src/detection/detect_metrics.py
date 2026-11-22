"""Đánh giá detection: Precision, Recall, F1, AP (giao thức YE-1 mục 4).

- IoU >= 0.5 mới tính là khớp. Ghép theo score giảm dần, mỗi GT chỉ được ghép tối đa 1 lần.
- Mặt invalid bị loại: prediction trùng mặt invalid thì bỏ qua, không tính là FP.
- AP: all-point interpolation (lấy precision lớn nhất ở bên phải rồi tính diện tích dưới đường).
- Kết quả chỉ so sánh được trong nhóm, không so với bảng xếp hạng WIDER chính thức.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import yaml

from src.detection.wider_data import DEFAULT_SIZE_GROUPS, image_key, load_ground_truth

# trạng thái của một prediction sau khi ghép với GT
TP, FP, IGNORE = 1, 0, -1


def box_iou(a, b):
    """Tính IoU giữa mọi cặp box xyxy của a và b, kết quả có shape (len(a), len(b))."""
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    # [:, None] và [None, :] để numpy tự tính cho mọi cặp (a_i, b_j) cùng lúc.
    # w, h là chiều rộng và cao của phần giao, clip về 0 khi hai box không chồng nhau
    w = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    h = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = w * h
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    # union = 0 (box suy biến) thì cho IoU = 0, và dùng maximum để tránh cảnh báo chia cho 0
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def match_image(boxes, scores, gt_boxes, gt_ignore, iou_thr=0.5, pred_ignore=None):
    """Ghép các prediction của một ảnh với GT.

    Trả về (scores, status) đã sắp theo score giảm dần.
    status: 1 = TP, 0 = FP, -1 = bỏ qua (không tính vào TP lẫn FP).
    """
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    scores = np.asarray(scores, float)
    # xử lý box có score cao trước, kind="stable" để score bằng nhau vẫn giữ thứ tự cũ
    order = np.argsort(-scores, kind="stable")
    boxes, scores = boxes[order], scores[order]
    out_of_range = np.zeros(len(scores), bool) if pred_ignore is None else np.asarray(pred_ignore)[order]
    ious = box_iou(boxes, gt_boxes)
    taken = np.zeros(len(gt_boxes), bool)
    status = np.zeros(len(scores), dtype=np.int8)
    for k in range(len(scores)):
        # ảnh không có GT nào thì ious có 0 cột, thay bằng mảng rỗng để khỏi lỗi
        row = ious[k] if ious.shape[1] else np.zeros(0)
        ok = row >= iou_thr
        # ưu tiên GT hợp lệ chưa ghép, IoU cao nhất
        cand = ok & ~gt_ignore & ~taken
        if cand.any():
            j = np.argmax(np.where(cand, row, -1))
            taken[j] = True
            status[k] = TP
        elif (ok & gt_ignore).any():
            status[k] = IGNORE  # trùng mặt invalid
        else:
            # không khớp GT nào. Nếu box nằm ngoài nhóm kích thước đang xét thì bỏ qua,
            # còn lại mới là FP thật
            status[k] = IGNORE if out_of_range[k] else FP
    return scores, status


def average_precision(precision, recall):
    """AP theo kiểu all-point: diện tích dưới đường precision-recall sau khi làm 'bao' precision."""
    # thêm điểm đầu (recall 0) và điểm cuối (recall 1) cho đường cong khép kín
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):  # precision envelope
        # đi từ phải sang trái, mỗi điểm lấy precision lớn nhất ở bên phải nó
        mpre[i] = max(mpre[i], mpre[i + 1])
    # chỉ cộng diện tích ở những chỗ recall thật sự tăng
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def prf(tp, fp, fn):
    """Tính (P, R, F1). Nếu không có prediction và cũng không có GT thì cả ba bằng 1."""
    if tp + fp == 0 and tp + fn == 0:
        return 1.0, 1.0, 1.0
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def _collect(preds, gt, ids, iou_thr, height_range):
    """Ghép prediction với GT cho mọi ảnh trong ids, gom lại thành score, status và tổng số GT hợp lệ.

    height_range là (thấp, cao) theo chiều cao box để xét riêng một nhóm kích thước, None là xét tất cả.
    """
    all_scores, all_status, n_gt = [], [], 0
    for i in ids:
        gt_boxes, invalid = gt[i]
        ignore = invalid.copy()
        if height_range:  # nhóm kích thước: GT ngoài nhóm coi như invalid
            h = gt_boxes[:, 3] - gt_boxes[:, 1]
            ignore |= (h < height_range[0]) | (h >= height_range[1])
        # chỉ đếm GT còn hợp lệ vào mẫu số của recall
        n_gt += int((~ignore).sum())
        faces = preds.get(i, [])
        boxes = np.asarray([f["bbox"] for f in faces], float).reshape(-1, 4)
        scores = np.asarray([f["score"] for f in faces], float)
        pred_ignore = None
        if height_range and len(boxes):
            # prediction ngoài nhóm kích thước cũng bỏ qua nếu không khớp GT nào,
            # để không bị tính FP oan khi đang xét nhóm khác
            ph = boxes[:, 3] - boxes[:, 1]
            pred_ignore = (ph < height_range[0]) | (ph >= height_range[1])
        s, st = match_image(boxes, scores, gt_boxes, ignore, iou_thr, pred_ignore)
        all_scores.append(s)
        all_status.append(st)
    if not all_scores:
        return np.zeros(0), np.zeros(0, dtype=np.int8), n_gt
    return np.concatenate(all_scores), np.concatenate(all_status), n_gt


def _summary(scores, status, n_gt, score_thr):
    """Từ score và status của mọi prediction, tính TP/FP/FN, P/R/F1, AP và điểm F1 tốt nhất."""
    # bỏ các prediction bị bỏ qua rồi sắp lại theo score giảm dần trên toàn tập
    keep = status != IGNORE
    s, st = scores[keep], status[keep]
    order = np.argsort(-s, kind="stable")
    s, st = s[order], st[order]
    # tp_cum, fp_cum: số TP và FP tích lũy khi hạ dần ngưỡng score, dùng để vẽ đường PR
    tp_cum, fp_cum = np.cumsum(st == TP), np.cumsum(st == FP)
    if len(s):
        last = np.r_[s[1:] != s[:-1], True]  # score trùng nhau thì lấy điểm cuối nhóm
        tp_c, fp_c, thr = tp_cum[last], fp_cum[last], s[last]
    else:
        tp_c = fp_c = thr = np.zeros(0)
    precision = tp_c / np.maximum(tp_c + fp_c, 1)
    recall = tp_c / n_gt if n_gt else np.zeros(len(tp_c))
    if n_gt == 0:
        # không có GT nào: không prediction nào thì AP = 1, có prediction (toàn FP) thì AP = 0
        ap = 1.0 if len(s) == 0 else 0.0
    else:
        ap = average_precision(precision, recall)
    # TP, FP tại ngưỡng score_thr được chọn, từ đó suy ra FN và P/R/F1
    sel = s >= score_thr
    tp, fp = int((st[sel] == TP).sum()), int((st[sel] == FP).sum())
    p, r, f1 = prf(tp, fp, n_gt - tp)
    out = {"n_gt": n_gt, "tp": tp, "fp": fp, "fn": n_gt - tp,
           "precision": p, "recall": r, "f1": f1, "ap": ap}
    if len(thr):  # score cho F1 cao nhất, dùng để tune ngưỡng Haar
        f1s = [prf(int(t), int(f), int(n_gt - t))[2] for t, f in zip(tp_c, fp_c)]
        b = int(np.argmax(f1s))
        out["best_f1"] = {"score_thr": float(thr[b]), "f1": f1s[b],
                          "precision": float(precision[b]), "recall": float(recall[b])}
    # đường cong PR giữ lại để vẽ hình nếu cần, lúc ghi JSON thì bỏ đi (xem main)
    out["_curve"] = (precision, recall)
    return out


def evaluate_detections(preds, gt, iou_thr=0.5, score_thr=0.5,
                        size_groups=DEFAULT_SIZE_GROUPS, require_full=True):
    """Đánh giá toàn bộ predictions so với GT.

    preds: {đường dẫn ảnh: [{bbox, score}]}. Trả {"all": ..., "small": ..., "medium": ..., "large": ...}.
    require_full=True thì mọi ảnh trong GT đều phải có prediction (kể cả danh sách rỗng).
    """
    # đổi khóa của preds về dạng '<event>/<tên file>' để khớp với khóa của GT
    preds = {image_key(k): v for k, v in preds.items()}
    unknown = sorted(set(preds) - set(gt))
    if unknown:
        raise ValueError(f"{len(unknown)} ảnh trong predictions không có nhãn: {unknown[:3]}")
    missing = set(gt) - set(preds)
    if missing and require_full:
        raise ValueError(f"Thiếu prediction cho {len(missing)} ảnh, ví dụ {sorted(missing)[:3]}")
    ids = sorted(preds)
    result = {"all": _summary(*_collect(preds, gt, ids, iou_thr, None), score_thr)}
    if size_groups:
        # chia theo chiều cao box: nhỏ < lo, vừa từ lo đến hi, lớn >= hi
        lo, hi = size_groups
        for name, rng in (("small", (0, lo)), ("medium", (lo, hi)), ("large", (hi, float("inf")))):
            result[name] = _summary(*_collect(preds, gt, ids, iou_thr, rng), score_thr)
    return result


def print_table(result):
    """In bảng kết quả theo nhóm kích thước ra màn hình."""
    print(f"{'nhóm':<8}{'GT':>7}{'TP':>7}{'FP':>7}{'FN':>7}{'P':>8}{'R':>8}{'F1':>8}{'AP':>8}")
    for name, r in result.items():
        print(f"{name:<8}{r['n_gt']:>7}{r['tp']:>7}{r['fp']:>7}{r['fn']:>7}"
              f"{r['precision']:>8.4f}{r['recall']:>8.4f}{r['f1']:>8.4f}{r['ap']:>8.4f}")


def main():
    """Chạy từ dòng lệnh: đọc predictions và GT, in bảng P/R/F1/AP, có thể ghi ra JSON."""
    p = argparse.ArgumentParser(description="Tính P/R/F1/AP cho một file predictions.")
    p.add_argument("--pred", required=True, help="JSON từ run_detect")
    p.add_argument("--gt", required=True, help="wider_face_val_bbx_gt.txt")
    p.add_argument("--detector", choices=("retinaface", "haar"), help="đọc score_thr từ --locked-config")
    p.add_argument("--locked-config", default="configs/detection_locked.yaml")
    p.add_argument("--score-thr", type=float, default=None)
    p.add_argument("--allow-partial", action="store_true", help="cho phép thiếu ảnh (chạy thử)")
    p.add_argument("--out", help="ghi kết quả ra JSON")
    a = p.parse_args()

    # ngưỡng score: ưu tiên --score-thr nếu có, không thì đọc từ config đã khóa theo --detector
    thr = a.score_thr
    if thr is None:
        if not a.detector:
            raise SystemExit("Cần --score-thr hoặc --detector (đọc từ config đã khóa).")
        with open(a.locked_config, encoding="utf-8") as f:
            thr = float(yaml.safe_load(f)[a.detector]["score_thr"])
    with open(a.pred, encoding="utf-8") as f:
        preds = json.load(f)
    result = evaluate_detections(preds, load_ground_truth(a.gt), score_thr=thr,
                                 require_full=not a.allow_partial)
    print(f"score_thr = {thr}")
    print_table(result)
    if a.detector == "haar":
        print("Lưu ý: AP của Haar chỉ để tham khảo, không so trực tiếp với AP của RetinaFace.")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            # bỏ _curve vì là mảng numpy, không ghi được ra JSON
            out = {k: {x: y for x, y in v.items() if x != "_curve"} for k, v in result.items()}
            # ghi kèm ngưỡng/detector để bảng báo cáo không phải đoán lại
            out["score_thr"] = thr
            if a.detector:
                out["detector"] = a.detector
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
