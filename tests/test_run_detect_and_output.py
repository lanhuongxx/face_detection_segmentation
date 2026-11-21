import json
import sys

import numpy as np

from src.detection import detect_metrics, run_detect


def test_retinaface_model_name_comes_from_config_and_cli(monkeypatch):
    seen = {}

    class Fake:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def detect(self, image):
            return []

    monkeypatch.setattr(run_detect, "RetinaFaceDetector", Fake)
    run_detect.create_detector("retinaface", {"retinaface": {"model_name": "mobile0.25"}})
    assert seen["model_name"] == "mobile0.25"
    run_detect.create_detector("retinaface", {})
    assert seen["model_name"] == "resnet50"  # mặc định cũ không đổi


def test_run_detection_model_name_override(monkeypatch, tmp_path):
    seen = {}

    class Fake:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def detect(self, image):
            return []

    monkeypatch.setattr(run_detect, "RetinaFaceDetector", Fake)
    import cv2
    cv2.imwrite(str(tmp_path / "a.jpg"), np.zeros((40, 40, 3), np.uint8))
    (tmp_path / "m.csv").write_text("image_path,split\na.jpg,test\n")
    cfg = tmp_path / "c.yaml"
    cfg.write_text("retinaface:\n  conf_thres: 0.5\n")
    run_detect.run_detection("retinaface", tmp_path / "m.csv", tmp_path / "o.json", cfg,
                             model_name="mobile0.25")
    assert seen["model_name"] == "mobile0.25"


def test_detect_metrics_json_records_threshold_and_detector(tmp_path, monkeypatch):
    gt = tmp_path / "gt.txt"
    gt.write_text("0--E/a.jpg\n1\n10 10 50 60 0 0 0 0 0 0\n")
    pred = tmp_path / "p.json"
    pred.write_text(json.dumps({"x/0--E/a.jpg": [{"bbox": [10, 10, 60, 70], "score": 0.9}]}))
    out = tmp_path / "o.json"
    monkeypatch.setattr(sys, "argv", ["m", "--pred", str(pred), "--gt", str(gt),
                                      "--score-thr", "0.5", "--detector", "retinaface",
                                      "--locked-config", str(tmp_path / "none.yaml"),
                                      "--out", str(out)])
    detect_metrics.main()
    data = json.loads(out.read_text())
    assert data["score_thr"] == 0.5 and data["detector"] == "retinaface"
    assert data["all"]["tp"] == 1
