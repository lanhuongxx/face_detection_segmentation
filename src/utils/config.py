"""Cấu hình mặc định của pipeline. File configs/pipeline.yaml (nếu có) sẽ ghi đè từng mục."""
from copy import deepcopy
from pathlib import Path

DEFAULT_CONFIG = {
    "device": "auto",                       # auto | cuda | cpu
    "preprocess": {"max_side": 4096, "max_file_mb": 10},
    "detection": {
        "model": "retinaface_mobile0.25",   # nhanh, đã dùng ở notebook. Đổi "retinaface_resnet50" nếu cần bắt mặt nhỏ tốt hơn
        "conf_threshold": 0.5,              # thấp hơn = ít bỏ sót mặt hơn
        "max_faces": 200,
    },
    "crop": {"margin": 0.25, "size": 256, "min_face_px": 20},
    "segmentation": {
        "weights": "models/unet_best.pt",
        "mask_threshold": 0.5,
        "min_mask_ratio": 0.30,             # mask phủ < 30% bbox thì coi là hỏng, dùng ellipse
        "batch_size": 16,
    },
    "privacy": {
        "mode": "blur",                     # blur | pixelate | solid
        "strength": 6,                      # 1-10
        "region": "mask",                   # mask | bbox | ellipse
        "dilate_px": 4,
        "feather_px": 3,
        "solid_color": [30, 30, 30],
    },
}


def _merge(base, extra):
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path="configs/pipeline.yaml"):
    """Trả về dict cấu hình: mặc định + giá trị từ file YAML (nếu file tồn tại)."""
    cfg = deepcopy(DEFAULT_CONFIG)
    path = Path(path) if path else None
    if path and path.is_file():
        import yaml
        with open(path, encoding="utf-8") as f:
            _merge(cfg, yaml.safe_load(f) or {})
    return cfg


def resolve_device(cfg):
    """'auto' -> cuda nếu có, ngược lại cpu."""
    device = cfg.get("device", "auto")
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
