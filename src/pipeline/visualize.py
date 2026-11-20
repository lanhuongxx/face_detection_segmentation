"""Module 6: vẽ hộp, vẽ mask, tạo thẻ khuôn mặt và báo cáo JSON."""
import numpy as np
import cv2

from src.pipeline.crop import crop_faces, square_crop_box

PALETTE = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
           (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212)]


def color_for(face_id):
    """Mỗi id một màu cố định."""
    return PALETTE[(face_id - 1) % len(PALETTE)]


def draw_boxes(image, faces):
    """Vẽ khung + nhãn '#id score' lên bản sao của ảnh."""
    out = image.copy()
    H, W = out.shape[:2]
    thick = max(2, int(round(min(H, W) / 300)))
    scale = max(0.4, min(H, W) / 1200)
    for face in faces:
        x1, y1, x2, y2 = face.bbox
        color = color_for(face.id)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
        label = f"#{face.id} {face.score:.2f}" + (" e" if face.status != "ok" else "")
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        ty = y1 - 4 if y1 - th - base - 4 > 0 else y2 + th + 4
        cv2.rectangle(out, (x1, ty - th - base), (x1 + tw + 4, ty + base), color, -1)
        cv2.putText(out, label, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def draw_masks(image, faces, instance_map):
    """Phủ màu mask từng mặt (mỗi mặt một màu) lên bản sao của ảnh."""
    if not faces:
        return image.copy()
    lut = np.zeros((int(instance_map.max()) + 1, 3), np.float32)
    for face in faces:
        if face.id < len(lut):
            lut[face.id] = color_for(face.id)
    painted = lut[instance_map]
    covered = (instance_map > 0)[..., None]
    mixed = image.astype(np.float32) * 0.5 + painted * 0.5
    return np.where(covered, mixed, image).astype(np.uint8)


def make_thumbnail(image, bbox, size=96, margin=0.15):
    """Ảnh crop vuông nhỏ quanh một khuôn mặt, dùng cho thẻ khuôn mặt."""
    return crop_faces(image, [square_crop_box(bbox, margin)], size)[0]


def build_report(result, params=None):
    """Báo cáo JSON (dict) theo mục 3.6 của file thiết kế."""
    faces = []
    for f in result.faces:
        faces.append({
            "id": f.id,
            "bbox": [int(v) for v in f.bbox],
            "size_px": [int(v) for v in f.size_px],
            "score": round(float(f.score), 4),
            "landmarks": None if f.landmarks is None else np.asarray(f.landmarks).round(1).tolist(),
            "crop_box": None if f.crop_box is None else [int(v) for v in f.crop_box],
            "mask_area_px": 0 if f.mask_local is None else int(f.mask_local.sum()),
            "status": f.status,
            "reason": f.reason,
            "blurred": bool(f.blurred),
            "identity": f.identity,                 # luôn null ở phiên bản hiện tại
            "identity_score": f.identity_score,
        })
    h, w = result.image.shape[:2]
    return {
        "source": {**result.meta.get("source", {}), "width": int(w), "height": int(h)},
        "params": params or result.meta.get("params", {}),
        "num_faces": len(faces),
        "faces": faces,
        "timing_ms": {k: round(float(v), 1) for k, v in result.timing_ms.items()},
        "device": result.meta.get("device"),
    }
