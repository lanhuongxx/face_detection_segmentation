"""Tổng hợp bảng và hình kết quả cho báo cáo (YE-4).

Mọi số liệu được tính lại từ các file log do ``evaluate.py`` (CSV theo ảnh),
``crop_robustness.py`` và ``detect_metrics`` (JSON) ghi ra, nên có nguồn truy vết.
Không nhập tay số vào bảng.

Ví dụ::

    python -m src.evaluation.report_tables \\
        --run BCE=outputs/eval/val_bce.csv --run Dice=outputs/eval/val_dice.csv \\
        --run BCE+Dice=outputs/eval/val_bce_dice.csv \\
        --detection-json outputs/detection/detection_comparison.json \\
        --crop-summary outputs/eval/crop_robustness/crop_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np

_METRICS = ("dice", "iou", "precision", "recall")


def load_per_image(path: Path) -> dict[str, np.ndarray]:
    """Đọc CSV của evaluate.py thành các mảng cột."""
    with path.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"CSV rỗng: {path}")
    needed = ("image_id", *_METRICS, "area_gt", "area_pred")
    missing = [c for c in needed if c not in rows[0]]
    if missing:
        raise ValueError(f"{path} thiếu cột {missing}; đây có phải CSV của evaluate.py không?")
    data = {c: np.array([r[c] for r in rows], dtype=object if c == "image_id" else np.float64)
            for c in needed}
    return data


def macro_stats(data: dict[str, np.ndarray]) -> dict[str, float]:
    """Số chính: thống kê trên metric từng ảnh."""
    out: dict[str, float] = {"n": float(len(data["dice"]))}
    for name in _METRICS:
        values = data[name]
        out[f"{name}_mean"] = float(values.mean())
        out[f"{name}_std"] = float(values.std())
        out[f"{name}_median"] = float(np.median(values))
    return out


def micro_stats(data: dict[str, np.ndarray]) -> dict[str, float]:
    """Số bổ sung: gộp TP/FP/FN toàn bộ pixel.

    TP của từng ảnh suy ra từ Dice: ``TP = Dice * (area_gt + area_pred) / 2``. Công thức
    đúng với mọi ảnh (cả khi mask rỗng thì TP = 0), nên không cần đọc lại mask.
    """
    area_gt, area_pred = data["area_gt"], data["area_pred"]
    tp = np.rint(data["dice"] * (area_gt + area_pred) / 2.0)
    tp_sum, gt_sum, pred_sum = tp.sum(), area_gt.sum(), area_pred.sum()
    fp, fn = pred_sum - tp_sum, gt_sum - tp_sum
    dice_den, iou_den = 2 * tp_sum + fp + fn, tp_sum + fp + fn
    return {
        "dice": 2 * tp_sum / dice_den if dice_den else 1.0,
        "iou": tp_sum / iou_den if iou_den else 1.0,
        "precision": tp_sum / pred_sum if pred_sum else (1.0 if gt_sum == 0 else 0.0),
        "recall": tp_sum / gt_sum if gt_sum else (1.0 if pred_sum == 0 else 0.0),
    }


def compare_runs(runs: dict[str, Path]) -> list[dict[str, float | str]]:
    rows = []
    for name, path in runs.items():
        data = load_per_image(path)
        macro, micro = macro_stats(data), micro_stats(data)
        rows.append({"run": name, "source": str(path), **macro,
                     **{f"micro_{k}": v for k, v in micro.items()}})
    return rows


def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def segmentation_markdown(rows: Sequence[dict], target: float | None) -> str:
    head = ("| Run | N | Dice (mean ± std) | Dice median | IoU (mean ± std) | "
            "Pixel P | Pixel R | Dice micro |")
    lines = [head, "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(
            f"| {r['run']} | {int(r['n'])} | {_fmt(r['dice_mean'])} ± {_fmt(r['dice_std'])} "
            f"| {_fmt(r['dice_median'])} | {_fmt(r['iou_mean'])} ± {_fmt(r['iou_std'])} "
            f"| {_fmt(r['precision_mean'])} | {_fmt(r['recall_mean'])} "
            f"| {_fmt(r['micro_dice'])} |"
        )
    text = "\n".join(lines)
    if target is not None:
        best = max(rows, key=lambda r: r["dice_mean"])
        verdict = "đạt" if best["dice_mean"] >= target else "chưa đạt"
        text += (f"\n\nMục tiêu thử nghiệm Dice ≥ {target}: run cao nhất ({best['run']}) "
                 f"{verdict} ({_fmt(best['dice_mean'])}). Mục tiêu này chỉ có ý nghĩa với "
                 "số trên tập test; số trên validation không dùng để kết luận đạt/chưa đạt.")
    return text


def load_detection_results(specs: Sequence[str]) -> dict[str, dict]:
    """Đọc kết quả detection từ nhiều file. Mỗi spec là ``PATH`` hoặc ``TEN=PATH[@ngưỡng]``.

    Hai định dạng đều được nhận:
    - của Uyên (``detect_metrics --out``): một detector mỗi file, khóa ``all/small/medium/large``;
      ngưỡng lấy từ khóa ``score_thr`` nếu có, hoặc ``@ngưỡng`` trong spec;
    - dạng gộp: ``{ten_detector: {score_thr, all, small, ...}}``.
    """
    merged: dict[str, dict] = {}
    for spec in specs:
        name = None
        thr = None
        text = spec
        if "=" in text:
            name, text = text.split("=", 1)
        if "@" in text:
            head, tail = text.rsplit("@", 1)
            try:
                thr = float(tail)
                text = head
            except ValueError:
                pass
        path = Path(text.strip())
        data = json.loads(path.read_text(encoding="utf-8"))
        if "all" in data and isinstance(data["all"], dict) and "precision" in data["all"]:
            label = (name or data.get("detector") or path.stem).strip()
            value = thr if thr is not None else data.get("score_thr")
            if value is None:
                raise ValueError(
                    f"{path}: thiếu ngưỡng score; chạy lại detect_metrics bản mới hoặc ghi "
                    f"{label}={path}@0.5")
            data = {**data, "score_thr": float(value)}
            data.pop("detector", None)
            data = {label: data}
        for label, result in data.items():
            if label in merged:
                raise ValueError(f"tên detector trùng: {label}")
            merged[label] = result
    return merged


def detection_markdown(source) -> str:
    """``source`` là một Path (định dạng gộp cũ) hoặc danh sách spec (xem load_detection_results)."""
    specs = [str(source)] if isinstance(source, (str, Path)) else list(source)
    results = load_detection_results(specs)
    lines = ["| Detector | Nhóm | GT | TP | FP | FN | P | R | F1 | AP |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, result in results.items():
        for group in ("all", "small", "medium", "large"):
            r = result[group]
            ap = "-" if r["ap"] is None else _fmt(r["ap"], 3)
            lines.append(
                f"| {name} (score ≥ {result['score_thr']:.3g}) | {group} | {r['n_gt']} "
                f"| {r['tp']} | {r['fp']} | {r['fn']} | {_fmt(r['precision'], 3)} "
                f"| {_fmt(r['recall'], 3)} | {_fmt(r['f1'], 3)} | {ap} |"
            )
    lines.append("\nAP của Haar chỉ để tham khảo (score khác thang RetinaFace); "
                 "so sánh chính là P/R/F1 tại ngưỡng đã khóa. Không so trực tiếp với "
                 "bảng xếp hạng WIDER FACE.")
    return "\n".join(lines)


def crop_markdown(path: Path) -> str:
    with path.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    lines = ["| Cách cắt | N | Miss | Box IoU | Dice | Giảm so với GT crop |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(
            f"| {r['setting']} | {r['n']} | {float(r['miss_rate']):.3f} "
            f"| {float(r['box_iou_mean']):.3f} | {float(r['dice_mean']):.4f} "
            f"| {float(r['dice_drop_vs_gt']):+.4f} |"
        )
    return "\n".join(lines)


def boxplot(runs: dict[str, Path], path: Path) -> None:
    """Hộp Dice theo từng run. Dùng backend không cần màn hình."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = [load_per_image(p)["dice"] for p in runs.values()]
    fig, ax = plt.subplots(figsize=(1.6 * len(values) + 2, 4))
    try:  # matplotlib >= 3.9 đổi tên labels -> tick_labels
        ax.boxplot(values, tick_labels=list(runs), showmeans=True)
    except TypeError:
        ax.boxplot(values, labels=list(runs), showmeans=True)
    ax.set_ylabel("Dice từng ảnh")
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Phân bố Dice theo run")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _parse_run(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("dùng dạng TEN=duong_dan.csv")
    name, path = text.split("=", 1)
    return name.strip(), Path(path.strip())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", type=_parse_run, default=[],
                        help="TEN=csv từ evaluate.py (lặp lại cho nhiều run)")
    parser.add_argument("--target", type=float, default=0.85,
                        help="Dice mục tiêu (phiếu đăng ký: 0.85)")
    parser.add_argument("--detection-json", action="append", default=[],
                        help="TEN=json[@ngưỡng] từ detect_metrics; lặp lại cho mỗi detector")
    parser.add_argument("--crop-summary", type=Path)
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/eval/report"))
    args = parser.parse_args(argv)
    if not (args.run or args.detection_json or args.crop_summary):
        parser.error("cần ít nhất một trong --run, --detection-json, --crop-summary")
    names = [n for n, _ in args.run]
    if len(set(names)) != len(names):
        parser.error("tên run bị trùng")

    sections = ["# Bảng kết quả (tự sinh từ log, không nhập tay)\n"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    try:
        if args.run:
            runs = dict(args.run)
            rows = compare_runs(runs)
            with (args.out_dir / "seg_comparison.csv").open(
                "w", newline="", encoding="utf-8"
            ) as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            boxplot(runs, args.out_dir / "dice_boxplot.png")
            sections += ["## Segmentation\n", segmentation_markdown(rows, args.target),
                         "\n![Dice boxplot](dice_boxplot.png)\n"]
        if args.detection_json:
            sections += ["## Detection (RetinaFace vs Haar)\n",
                         detection_markdown(args.detection_json), ""]
        if args.crop_summary:
            sections += ["## GT crop vs predicted crop\n", crop_markdown(args.crop_summary), ""]
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    out = args.out_dir / "results_tables.md"
    out.write_text("\n".join(sections), encoding="utf-8")
    print(out.read_text(encoding="utf-8"))
    print(f"Đã ghi: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
