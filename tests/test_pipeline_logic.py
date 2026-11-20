"""Kiểm thử logic pipeline bằng detector/segmenter GIẢ (không cần GPU, torch hay facexlib).
Chạy:  python tests/test_pipeline_logic.py      (hoặc: python -m pytest tests)"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.inference import FacePipeline
from src.pipeline.mask_mapping import clean_mask
from src.pipeline.preprocess_input import (CorruptImage, FileTooLarge, UnsupportedFormat, limit_size,
                                           load_image)
from src.pipeline.schemas import Detection
from src.utils.config import load_config


class FakeDetector:
    def __init__(self, boxes_scores):
        self.items = boxes_scores

    def detect(self, image, conf_threshold=0.5):
        return [Detection(b, s) for b, s in self.items if s >= conf_threshold]


class FakeSegmenter:
    """Trả về mặt nạ hình elip nằm giữa crop (giả lập U-Net chạy đúng)."""
    def __init__(self, empty=False):
        self.empty = empty

    def predict(self, crops):
        n, s = crops.shape[0], crops.shape[1]
        out = np.zeros((n, s, s), np.float32)
        if not self.empty:
            for i in range(n):
                cv2.ellipse(out[i], (s // 2, s // 2), (int(s * 0.28), int(s * 0.34)), 0, 0, 360, 1.0, -1)
        return out


def make_image(h=480, w=640, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def make_pipeline(items, empty_seg=False):
    return FacePipeline(load_config(None), FakeDetector(items), FakeSegmenter(empty_seg))


def test_no_faces():
    img = make_image()
    pipe = make_pipeline([])
    res = pipe.analyze(img)
    assert res.faces == [] and res.instance_map.max() == 0
    out = pipe.apply_privacy(res, [])
    assert np.array_equal(out, img), "Ảnh không có mặt phải giữ nguyên"
    assert pipe.report(res)["num_faces"] == 0


def test_statuses_and_ids():
    img = make_image()
    # 3 mặt: bình thường, bình thường, mặt rất nhỏ. Đưa vào lộn xộn để kiểm tra sắp theo x1.
    pipe = make_pipeline([((400, 100, 500, 220), 0.9), ((100, 100, 200, 220), 0.95), ((300, 300, 314, 316), 0.7)])
    res = pipe.analyze(img)
    assert [f.id for f in res.faces] == [1, 2, 3]
    assert [f.bbox[0] for f in res.faces] == sorted(f.bbox[0] for f in res.faces)
    small = [f for f in res.faces if f.reason == "too_small"]
    assert len(small) == 1 and small[0].status == "fallback_ellipse"
    ok = [f for f in res.faces if f.status == "ok"]
    assert len(ok) == 2 and all(f.mask_local.any() for f in ok)
    assert all(f.thumbnail.shape == (96, 96, 3) for f in res.faces)
    assert set(np.unique(res.instance_map)) == {0, 1, 2, 3}


def test_bad_mask_fallback():
    pipe = make_pipeline([((100, 100, 200, 220), 0.9)], empty_seg=True)   # U-Net "hỏng": mask rỗng
    res = pipe.analyze(make_image())
    f = res.faces[0]
    assert f.status == "fallback_ellipse" and f.reason == "bad_mask" and f.mask_local.any()


def test_face_at_image_edge():
    img = make_image()
    pipe = make_pipeline([((0, 0, 90, 110), 0.9), ((560, 380, 640, 480), 0.9)])   # sát góc: khung crop vượt biên
    res = pipe.analyze(img)
    assert len(res.faces) == 2 and all(f.status == "ok" for f in res.faces)
    for f in res.faces:
        assert f.crop_box[0] < 0 or f.crop_box[1] < 0 or f.crop_box[2] > 640 or f.crop_box[3] > 480
        ids = res.instance_map == f.id
        assert ids.any()
    out = pipe.apply_privacy(res, [1, 2])
    assert out.shape == img.shape and out.dtype == np.uint8


def test_privacy_only_changes_selected():
    img = make_image()
    pipe = make_pipeline([((100, 100, 200, 220), 0.9), ((400, 100, 500, 220), 0.9)])
    res = pipe.analyze(img)
    out = pipe.apply_privacy(res, [1])                     # chỉ che mặt 1
    diff = np.any(out != img, axis=2)
    assert diff[100:220, 100:200].any(), "Mặt được chọn phải bị thay đổi"
    assert not diff[:, 300:].any(), "Mặt không chọn và nền xa phải giữ nguyên từng pixel"
    assert res.faces[0].blurred and not res.faces[1].blurred
    for mode in ("blur", "pixelate", "solid"):
        for region in ("mask", "bbox", "ellipse"):
            o = pipe.apply_privacy(res, [1, 2], mode=mode, strength=8, region=region, store=False)
            assert np.any(o != img)


def test_blur_destroys_detail():
    """Che xong, vùng mặt phải 'mịn' hơn hẳn (ít chi tiết) so với ảnh gốc nhiễu."""
    img = make_image()
    pipe = make_pipeline([((200, 100, 320, 240), 0.9)])
    res = pipe.analyze(img)
    out = pipe.apply_privacy(res, [1], mode="blur", strength=6)
    inside = (slice(130, 210), slice(230, 290))
    assert out[inside].std() < 0.5 * img[inside].std()


def test_overlap_and_instance_map():
    pipe = make_pipeline([((100, 100, 220, 240), 0.99), ((160, 110, 280, 250), 0.6)])
    res = pipe.analyze(make_image())
    assert set(np.unique(res.instance_map)) <= {0, 1, 2}
    out = pipe.apply_privacy(res, [1, 2])
    assert out.shape == res.image.shape


def test_report_json():
    pipe = make_pipeline([((100, 100, 200, 220), 0.9), ((300, 300, 314, 316), 0.7)])
    res = pipe.analyze(make_image(), source={"type": "image", "filename": "a.jpg"})
    pipe.apply_privacy(res, [1])
    rep = pipe.report(res)
    text = json.dumps(rep)                                  # phải serialize được
    assert rep["num_faces"] == 2 and rep["source"]["filename"] == "a.jpg"
    assert {"id", "bbox", "size_px", "score", "status", "reason", "blurred", "identity"} <= set(rep["faces"][0])
    assert rep["faces"][0]["identity"] is None and rep["faces"][0]["blurred"] is True
    assert rep["faces"][1]["blurred"] is False and rep["faces"][1]["status"] == "fallback_ellipse"
    assert {"detect", "segment", "total", "filter"} <= set(rep["timing_ms"]) and len(text) > 0


def test_large_image_is_resized():
    cfg = load_config(None)
    cfg["preprocess"]["max_side"] = 400
    pipe = FacePipeline(cfg, FakeDetector([((50, 50, 150, 160), 0.9)]), FakeSegmenter())
    res = pipe.analyze(make_image(480, 640))
    assert max(res.image.shape[:2]) == 400 and abs(res.meta["scale"] - 400 / 640) < 1e-6
    small, s = limit_size(make_image(100, 100), 400)
    assert s == 1.0 and small.shape[:2] == (100, 100)


def test_conf_threshold_filters():
    pipe = make_pipeline([((100, 100, 200, 220), 0.9), ((300, 100, 400, 220), 0.3)])
    assert len(pipe.analyze(make_image(), conf_threshold=0.5).faces) == 1
    assert len(pipe.analyze(make_image(), conf_threshold=0.2).faces) == 2


def test_clean_mask_fills_holes_and_drops_islands():
    m = np.zeros((100, 100), bool)
    cv2.ellipse(m.view(np.uint8), (50, 50), (30, 40), 0, 0, 360, 1, -1)
    m[40:60, 30:45] = False                                 # lỗ thủng
    m[2:8, 2:8] = True                                      # mảng lẻ ngoài nền
    c = clean_mask(m)
    assert c[50, 38] and not c[4, 4]


def test_load_image_errors(tmp_path=None):
    import io
    from PIL import Image
    buf = io.BytesIO(); Image.fromarray(make_image(50, 60)).save(buf, "PNG")
    arr, meta = load_image(buf.getvalue())
    assert arr.shape == (50, 60, 3)
    try:
        load_image(b"khong phai anh"); assert False
    except (CorruptImage, UnsupportedFormat):
        pass
    try:
        load_image(b"x" * (11 * 1024 * 1024)); assert False
    except FileTooLarge:
        pass
    gif = io.BytesIO(); Image.fromarray(make_image(20, 20)).save(gif, "BMP")
    try:
        load_image(gif.getvalue()); assert False
    except UnsupportedFormat:
        pass


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn(); print("OK  ", name)
        except Exception as error:
            failed += 1; print("LỖI ", name, "->", type(error).__name__, error)
    print(f"\n{len(tests) - failed}/{len(tests)} test đạt")
    sys.exit(1 if failed else 0)
