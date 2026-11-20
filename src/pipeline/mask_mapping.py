"""Module 4: đưa mask 256x256 về đúng tọa độ ảnh gốc + các hàm xử lý mask."""
import cv2
import numpy as np


def clean_mask(mask):
    """mask bool (H,W). Giữ mảnh liên thông lớn nhất và lấp lỗ bên trong."""
    m = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return mask.astype(bool)
    m = (labels == 1 + stats[1:, cv2.CC_STAT_AREA].argmax()).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(m)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    return filled.astype(bool)


def prob_to_local_mask(prob, box, threshold):
    """prob (S,S) float -> mask uint8 {0,1} kích thước đúng bằng khung box (x1,y1,x2,y2).
    Resize xác suất rồi mới lấy ngưỡng để viền mượt hơn."""
    w, h = box[2] - box[0], box[3] - box[1]
    resized = cv2.resize(prob.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    return clean_mask(resized >= threshold).astype(np.uint8)


def ellipse_mask(bbox):
    """Mặt nạ hình elip nội tiếp trong bbox (dùng khi mặt quá nhỏ hoặc mask hỏng).
    Trả về (mask uint8 (h,w), khung đặt = bbox)."""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    mask = np.zeros((h, w), np.uint8)
    cv2.ellipse(mask, (w // 2, h // 2), (max(1, w // 2), max(1, h // 2)), 0, 0, 360, 1, -1)
    return mask, (x1, y1, x2, y2)


def _slices(dst_shape, src_shape, x, y):
    H, W = dst_shape[:2]
    h, w = src_shape[:2]
    dx1, dy1, dx2, dy2 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if dx2 <= dx1 or dy2 <= dy1:
        return None
    return (slice(dy1, dy2), slice(dx1, dx2)), (slice(dy1 - y, dy2 - y), slice(dx1 - x, dx2 - x))


def paste_mask(dst, src, x, y, value=None):
    """Đặt mask src vào dst tại (x,y), tự cắt phần vượt biên.
    value=None: OR (lấy max). value=k: gán k cho các pixel mask của src."""
    parts = _slices(dst.shape, src.shape, x, y)
    if parts is None:
        return
    d_sl, s_sl = parts
    if value is None:
        dst[d_sl] = np.maximum(dst[d_sl], src[s_sl].astype(dst.dtype))
    else:
        dst[d_sl][src[s_sl] > 0] = value


def mask_quality_ratio(mask_local, mask_box, bbox):
    """Phần mask nằm trong bbox / diện tích bbox. Thấp = mask hỏng."""
    ix1, iy1 = max(bbox[0], mask_box[0]), max(bbox[1], mask_box[1])
    ix2, iy2 = min(bbox[2], mask_box[2]), min(bbox[3], mask_box[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    sub = mask_local[iy1 - mask_box[1]:iy2 - mask_box[1], ix1 - mask_box[0]:ix2 - mask_box[0]]
    area = max((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), 1)
    return float(sub.sum()) / area


def build_instance_map(shape, faces):
    """Bản đồ (H,W) uint16: pixel = id của mặt. Chồng lấn: mặt có score cao hơn thắng."""
    instance = np.zeros(shape[:2], np.uint16)
    for face in sorted(faces, key=lambda f: f.score):          # vẽ score thấp trước, cao sau (đè lên)
        if face.mask_local is not None:
            paste_mask(instance, face.mask_local, face.mask_box[0], face.mask_box[1], value=face.id)
    return instance
