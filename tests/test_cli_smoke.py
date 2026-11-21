"""Chạy thật hai CLI (evaluate_pipeline, crop_robustness) với engine giả để bắt lỗi tham số."""

import csv

import cv2
import numpy as np
from PIL import Image

from src.evaluation import crop_robustness, evaluate_pipeline, pipeline_adapter
from src.evaluation.pipeline_adapter import MaskOutput


class FakeEngine:
    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def _out(shape, box):
        mask = np.zeros(shape, bool)
        if box is not None:
            x1, y1, x2, y2 = (int(v) for v in box)
            mask[y1:y2, x1:x2] = True
        return MaskOutput(mask, box, 0 if box is None else 1, False)

    def mask_from_boxes(self, image, boxes):
        return self._out(image.shape[:2], boxes[0])

    def mask_from_detector(self, image):
        return self._out(image.shape[:2], (60, 55, 140, 145))


def make_root(tmp_path, n=20):
    (tmp_path / "CelebA-HQ-img").mkdir()
    (tmp_path / "CelebAMask-HQ-mask-anno").mkdir()
    skin = np.zeros((200, 200), np.uint8)
    cv2.ellipse(skin, (100, 100), (40, 48), 0, 0, 360, 255, -1)
    for i in range(n):
        Image.fromarray(np.full((200, 200, 3), 90, np.uint8)).save(tmp_path / "CelebA-HQ-img" / f"{i}.jpg")
        Image.fromarray(skin).save(tmp_path / "CelebAMask-HQ-mask-anno" / f"{i:05d}_skin.png")
    return tmp_path


def test_evaluate_pipeline_cli_writes_csvs_and_masks(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_adapter, "PipelineEngine", FakeEngine)
    root, out = make_root(tmp_path), tmp_path / "out"
    code = evaluate_pipeline.main(["--data_root", str(root), "--split", "test", "--limit", "2",
                                   "--out_dir", str(out), "--save_masks"])
    assert code == 0
    for name in ("orig_clean.csv", "crop256_clean.csv", "summary.json"):
        assert (out / name).is_file()
    with open(out / "orig_clean.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2 and float(rows[0]["dice"]) > 0.7
    assert len(list((out / "pred").glob("*.png"))) == 2


def test_crop_robustness_cli_runs_with_only_required_args(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_adapter, "PipelineEngine", FakeEngine)
    root, out = make_root(tmp_path), tmp_path / "crop"
    code = crop_robustness.main(["--data_root", str(root), "--split", "test", "--limit", "2",
                                 "--out_dir", str(out)])
    assert code == 0
    with open(out / "crop_summary.csv") as f:
        settings = [r["setting"] for r in csv.DictReader(f)]
    assert settings[:2] == ["gt_crop", "pred_crop"] and len(settings) == 7


def test_documented_chain_error_analysis_finds_unpadded_celeba_names(tmp_path, monkeypatch):
    """Ảnh '0.jpg' nhưng CSV/mask '00000': lỗi này từng làm error_analysis dừng ngay."""
    from src.evaluation import error_analysis, report_tables

    monkeypatch.setattr(pipeline_adapter, "PipelineEngine", FakeEngine)
    root, out = make_root(tmp_path), tmp_path / "o"
    evaluate_pipeline.main(["--data_root", str(root), "--split", "test",
                            "--out_dir", str(out), "--save_masks"])
    assert error_analysis.main(["--metrics", str(out / "orig_clean.csv"),
                                "--image_dir", str(root / "CelebA-HQ-img"),
                                "--gt_dir", str(out / "gt_clean"), "--pred_dir", str(out / "pred"),
                                "--out_dir", str(tmp_path / "ea")]) == 0
    assert report_tables.main(["--run", f"Test={out / 'crop256_clean.csv'}",
                               "--out_dir", str(tmp_path / "rep")]) == 0
