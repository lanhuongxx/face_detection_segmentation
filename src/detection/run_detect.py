"""Chạy RetinaFace hoặc Haar trên cùng manifest và ghi predictions JSON."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import yaml

from src.detection.haar_baseline import HaarFaceDetector
from src.detection.retinaface import RetinaFaceDetector


_DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[2] / "configs" / "detection.yaml"
)


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Đọc YAML cấu hình và báo lỗi nếu nội dung không phải mapping."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Cấu hình phải là YAML mapping: {path}")
    return config


def load_manifest(
    manifest_path: str | Path, split: str | None = None
) -> list[tuple[str, Path]]:
    """Đọc các cột image_path, split; path tương đối tính từ thư mục CSV."""
    path = Path(manifest_path)
    rows: list[tuple[str, Path]] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        headers = set(reader.fieldnames or [])
        missing = {"image_path", "split"} - headers
        if missing:
            raise ValueError(
                "Manifest thiếu cột bắt buộc: " + ", ".join(sorted(missing))
            )

        for line_number, row in enumerate(reader, start=2):
            image_id = (row.get("image_path") or "").strip()
            row_split = (row.get("split") or "").strip()
            if not image_id or not row_split:
                raise ValueError(
                    f"Manifest có image_path/split rỗng tại dòng {line_number}."
                )
            if split is not None and row_split != split:
                continue
            if image_id in seen_ids:
                raise ValueError(
                    f"image_path trùng trong manifest tại dòng {line_number}: "
                    f"{image_id}"
                )
            seen_ids.add(image_id)
            image_path = Path(image_id)
            if not image_path.is_absolute():
                image_path = path.parent / image_path
            rows.append((image_id, image_path))

    if not rows:
        suffix = f" cho split '{split}'" if split is not None else ""
        raise ValueError(f"Manifest không có ảnh phù hợp{suffix}: {path}")
    return rows


def create_detector(
    detector_name: str, config: dict[str, Any]
) -> HaarFaceDetector | RetinaFaceDetector:
    """Tạo detector theo config chung để hai backend dùng cùng CLI."""
    if detector_name == "haar":
        settings = config.get("haar", {})
        if not isinstance(settings, dict):
            raise ValueError("Mục 'haar' trong config phải là mapping.")
        min_size = settings.get("min_size", [30, 30])
        if not isinstance(min_size, (list, tuple)) or len(min_size) != 2:
            raise ValueError("haar.min_size phải gồm width và height.")
        return HaarFaceDetector(
            scale_factor=settings.get("scale_factor", 1.1),
            min_neighbors=settings.get("min_neighbors", 5),
            min_size=tuple(min_size),
            equalize_hist=settings.get("equalize_hist", False),
        )
    if detector_name == "retinaface":
        settings = config.get("retinaface", {})
        if not isinstance(settings, dict):
            raise ValueError("Mục 'retinaface' trong config phải là mapping.")
        return RetinaFaceDetector(
            device=settings.get("device", "auto"),
            conf_thres=settings.get("conf_thres", 0.5),
            nms_thres=settings.get("nms_thres", 0.4),
        )
    raise ValueError(f"Detector không được hỗ trợ: {detector_name}")


def run_detection(
    detector_name: str,
    manifest_path: str | Path,
    output_path: str | Path,
    config_path: str | Path = _DEFAULT_CONFIG,
    split: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Chạy detector cho manifest rồi ghi bbox/score theo image_path."""
    rows = load_manifest(manifest_path, split=split)
    config = load_config(config_path)
    detector = create_detector(detector_name, config)
    predictions: dict[str, list[dict[str, Any]]] = {}

    for image_id, image_path in rows:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            if not image_path.is_file():
                raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")
            raise ValueError(f"Không giải mã được ảnh: {image_path}")
        faces = detector.detect(image)
        predictions[image_id] = [
            {"bbox": face["bbox"], "score": float(face["score"])}
            for face in faces
        ]

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as file:
        json.dump(predictions, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return predictions


def main(argv: list[str] | None = None) -> None:
    """Phân tích tham số CLI và chạy inference trên manifest."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="backslashreplace")

    parser = argparse.ArgumentParser(
        description="Chạy face detector trên cùng một CSV manifest."
    )
    parser.add_argument(
        "--detector",
        required=True,
        choices=("retinaface", "haar"),
        help="Backend phát hiện khuôn mặt.",
    )
    parser.add_argument("--manifest", required=True, help="CSV image_path,split.")
    parser.add_argument("--out", required=True, help="JSON predictions đầu ra.")
    parser.add_argument(
        "--config",
        default=str(_DEFAULT_CONFIG),
        help=f"YAML cấu hình (mặc định: {_DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Chỉ chạy split này; mặc định dùng mọi dòng trong manifest.",
    )
    args = parser.parse_args(argv)
    predictions = run_detection(
        detector_name=args.detector,
        manifest_path=args.manifest,
        output_path=args.out,
        config_path=args.config,
        split=args.split,
    )
    print(
        f"Đã xử lý {len(predictions)} ảnh bằng {args.detector}; "
        f"predictions: {args.out}"
    )


if __name__ == "__main__":
    main()
