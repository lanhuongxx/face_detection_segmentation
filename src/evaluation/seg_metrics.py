"""Các hàm tính metric segmentation (Dice, IoU, precision, recall) cho mask nhị phân.

Dùng được cho cả mảng NumPy lẫn tensor PyTorch, nên có thể gọi lúc validate trong
lúc train hoặc lúc đánh giá offline. Mask đầu vào có shape (H, W) cho một ảnh
hoặc (N, 1, H, W) cho một batch.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

try:
    import torch
except ImportError:  # máy nào chưa cài torch thì vẫn dùng được bản NumPy
    torch = None

# Ba cách gộp kết quả:
# - per_image: giữ riêng từng ảnh
# - mean: trung bình điểm của các ảnh (macro)
# - global: cộng TP/FP/FN của cả tập rồi mới tính điểm (micro)
Reduction = Literal["per_image", "mean", "global"]
_Array = Any


def _is_tensor(value: Any) -> bool:
    """Kiểm tra có phải torch.Tensor không (trả False luôn nếu chưa cài torch)."""
    return torch is not None and isinstance(value, torch.Tensor)


def _prepare_pair(pred: _Array, gt: _Array) -> tuple[_Array, _Array, bool]:
    """Kiểm tra hai mask hợp lệ, đưa về dạng batch (N, H, W) và đổi sang bool.

    Trả thêm một cờ cho biết đang dùng torch hay numpy để các hàm sau chọn đúng nhánh.
    """
    use_torch = _is_tensor(pred) or _is_tensor(gt)

    if use_torch:
        # chỉ cần một trong hai là tensor thì ép cả hai thành tensor trên cùng device
        tensors = [value for value in (pred, gt) if _is_tensor(value)]
        device = tensors[0].device
        if any(value.device != device for value in tensors[1:]):
            raise ValueError("pred and gt tensors must be on the same device")
        # detach để tính metric lúc validate không kéo theo graph của autograd
        pred_array = torch.as_tensor(pred, device=device).detach()
        gt_array = torch.as_tensor(gt, device=device).detach()
    else:
        pred_array = np.asarray(pred)
        gt_array = np.asarray(gt)

    if tuple(pred_array.shape) != tuple(gt_array.shape):
        raise ValueError(
            f"pred and gt must have the same shape; got "
            f"{tuple(pred_array.shape)} and {tuple(gt_array.shape)}"
        )

    shape = tuple(pred_array.shape)
    if len(shape) == 2:
        # một ảnh lẻ (H, W): thêm trục batch để dùng chung code với trường hợp nhiều ảnh
        if shape[0] == 0 or shape[1] == 0:
            raise ValueError("each mask must contain at least one pixel")
        pred_array = pred_array[None, ...]
        gt_array = gt_array[None, ...]
    elif len(shape) == 4 and shape[1] == 1:
        # batch dạng (N, 1, H, W): bỏ trục channel (luôn bằng 1) để còn (N, H, W)
        if shape[0] == 0 or shape[2] == 0 or shape[3] == 0:
            raise ValueError("batch masks must contain images with at least one pixel")
        pred_array = pred_array[:, 0, ...]
        gt_array = gt_array[:, 0, ...]
    else:
        raise ValueError(
            "masks must have shape (H, W) or (N, 1, H, W); "
            "if a mask has shape (N, H, W), use unsqueeze(1) or "
            "mask[:, None] to make it (N, 1, H, W); "
            f"got {shape}"
        )

    return _binarize(pred_array), _binarize(gt_array), use_torch


def _binarize(mask: _Array) -> _Array:
    """Đổi mask về bool: xác suất (0..1) lấy ngưỡng 0.5, mask ảnh (0/255) lấy ngưỡng 127."""
    if torch is not None and isinstance(mask, torch.Tensor):
        if mask.dtype == torch.bool:
            # đã là bool thì không cần so ngưỡng nữa
            return mask
        if not (
            torch.is_floating_point(mask)
            or mask.dtype
            in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
        ):
            raise ValueError(f"mask dtype must be numeric or bool, got {mask.dtype}")
        # NaN hoặc Inf thì so ngưỡng ra kết quả vô nghĩa, báo lỗi luôn cho dễ tìm
        if not bool(torch.isfinite(mask).all().item()):
            raise ValueError("masks must contain only finite values")
        if torch.is_floating_point(mask) and mask.numel() and bool(
            (mask.min() < 0).item()
        ):
            # giá trị âm gần như chắc chắn là logit chưa qua sigmoid
            raise ValueError(
                "float mask has negative values; looks like logits, apply sigmoid first"
            )
        # nếu giá trị nằm trong [0, 1] thì coi là xác suất (ngưỡng 0.5),
        # còn lớn hơn 1 thì coi là mask ảnh 0/255 (ngưỡng 127)
        threshold = (
            0.5
            if mask.numel() == 0 or bool(mask.max().item() <= 1)
            else 127
        )
        return mask > threshold

    if not (
        np.issubdtype(mask.dtype, np.integer)
        or np.issubdtype(mask.dtype, np.floating)
        or np.issubdtype(mask.dtype, np.bool_)
    ):
        raise ValueError(f"mask dtype must be numeric or bool, got {mask.dtype}")
    if not np.isfinite(mask).all():
        raise ValueError("masks must contain only finite values")
    if np.issubdtype(mask.dtype, np.floating) and mask.size and np.min(mask) < 0:
        raise ValueError(
            "float mask has negative values; looks like logits, apply sigmoid first"
        )
    # nhánh NumPy dùng đúng quy tắc ngưỡng như nhánh torch ở trên
    threshold = 0.5 if mask.size == 0 or np.max(mask) <= 1 else 127
    return mask > threshold


def _counts(pred: _Array, gt: _Array, use_torch: bool) -> tuple[_Array, ...]:
    """Đếm TP, FP, FN, TN riêng cho từng ảnh trong batch (hai mask đã là bool)."""
    if use_torch:
        dims = (1, 2)
        # TP = pixel cả hai cùng bằng 1. FP và FN suy ra từ diện tích mỗi mask trừ đi TP
        tp = (pred & gt).sum(dim=dims, dtype=torch.int64)
        pred_area = pred.sum(dim=dims, dtype=torch.int64)
        gt_area = gt.sum(dim=dims, dtype=torch.int64)
        fp = pred_area - tp
        fn = gt_area - tp
        # TN là phần còn lại của ảnh sau khi trừ TP, FP, FN
        total = pred.shape[1] * pred.shape[2]
        tn = total - tp - fp - fn
    else:
        dims = (1, 2)
        # nhánh NumPy tính y hệt, axis 1 và 2 là chiều cao và chiều rộng
        tp = np.count_nonzero(pred & gt, axis=dims)
        pred_area = np.count_nonzero(pred, axis=dims)
        gt_area = np.count_nonzero(gt, axis=dims)
        fp = pred_area - tp
        fn = gt_area - tp
        total = pred.shape[1] * pred.shape[2]
        tn = total - tp - fp - fn
    return tp, fp, fn, tn


def _reduce(value: _Array, reduction: Reduction, use_torch: bool) -> _Array:
    """Gộp một mảng giá trị theo kiểu reduction đã chọn."""
    if reduction == "per_image":
        return value
    if reduction == "mean":
        # đổi sang float trước, vì mảng đếm là số nguyên nên mean thẳng sẽ bị cụt
        return value.to(torch.float64).mean() if use_torch else value.mean()
    if reduction == "global":
        # với số đếm thì gộp toàn cục chính là cộng lại
        return value.sum()
    raise ValueError(
        "reduction must be 'per_image', 'mean', or 'global'; "
        f"got {reduction!r}"
    )


def _reduce_counts(
    counts: tuple[_Array, ...], reduction: Reduction, use_torch: bool
) -> tuple[_Array, ...]:
    """Áp dụng reduction lên từng loại số đếm TP, FP, FN, TN."""
    return tuple(_reduce(count, reduction, use_torch) for count in counts)


def _metric_counts(
    counts: tuple[_Array, ...], reduction: Reduction
) -> tuple[_Array, ...]:
    """Chọn số đếm để đưa vào công thức metric (với global thì cộng dồn trước)."""
    if reduction == "global":
        # micro: phải cộng TP/FP/FN của cả tập rồi mới chia,
        # không được cộng điểm Dice của từng ảnh
        return tuple(count.sum() for count in counts)
    if reduction not in ("per_image", "mean"):
        raise ValueError(
            "reduction must be 'per_image', 'mean', or 'global'; "
            f"got {reduction!r}"
        )
    return counts


def _reduce_metric(value: _Array, reduction: Reduction, use_torch: bool) -> _Array:
    """Điểm đã tính theo từng ảnh, chỉ cần lấy trung bình nếu chọn mean."""
    return _reduce(value, "mean", use_torch) if reduction == "mean" else value


def _ratio(
    numerator: _Array,
    denominator: _Array,
    zero_value: float,
    use_torch: bool,
) -> _Array:
    """Chia an toàn: chỗ nào mẫu số bằng 0 thì trả zero_value thay vì lỗi chia cho 0."""
    if use_torch:
        numerator = numerator.to(torch.float64)
        denominator = denominator.to(torch.float64)
        # clamp để phép chia không bao giờ chia cho 0, rồi torch.where chọn lại giá trị đúng
        safe_denominator = denominator.clamp(min=1)
        fallback = torch.full_like(numerator, zero_value)
        return torch.where(denominator > 0, numerator / safe_denominator, fallback)
    numerator = numerator.astype(np.float64, copy=False)
    denominator = denominator.astype(np.float64, copy=False)
    result = np.full(
        np.broadcast_shapes(numerator.shape, denominator.shape),
        zero_value,
        dtype=np.float64,
    )
    # chỉ chia ở những chỗ mẫu số > 0, các chỗ còn lại giữ nguyên zero_value
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def confusion_counts(
    pred: _Array,
    gt: _Array,
    reduction: Reduction = "per_image",
) -> tuple[_Array, _Array, _Array, _Array]:
    """Trả về (TP, FP, FN, TN). Mặc định là số đếm riêng của từng ảnh."""
    pred_binary, gt_binary, use_torch = _prepare_pair(pred, gt)
    # với số đếm thì global cộng thẳng được, khác với metric (xem _metric_counts)
    return _reduce_counts(
        _counts(pred_binary, gt_binary, use_torch), reduction, use_torch
    )


def dice_score(
    pred: _Array,
    gt: _Array,
    reduction: Reduction = "mean",
) -> _Array:
    """Dice = 2TP / (2TP + FP + FN). Hai mask cùng rỗng thì quy ước Dice = 1."""
    pred_binary, gt_binary, use_torch = _prepare_pair(pred, gt)
    # nếu là global thì gộp số đếm trước, sau đó mới tính ra Dice micro
    tp, fp, fn, _ = _metric_counts(
        _counts(pred_binary, gt_binary, use_torch), reduction
    )
    numerator = tp * 2
    denominator = numerator + fp + fn
    score = _ratio(numerator, denominator, 1.0, use_torch)
    return _reduce_metric(score, reduction, use_torch)


def iou_score(
    pred: _Array,
    gt: _Array,
    reduction: Reduction = "mean",
) -> _Array:
    """IoU = TP / (TP + FP + FN). Hai mask cùng rỗng thì quy ước IoU = 1."""
    pred_binary, gt_binary, use_torch = _prepare_pair(pred, gt)
    tp, fp, fn, _ = _metric_counts(
        _counts(pred_binary, gt_binary, use_torch), reduction
    )
    score = _ratio(tp, tp + fp + fn, 1.0, use_torch)
    return _reduce_metric(score, reduction, use_torch)


def pixel_precision_recall(
    pred: _Array,
    gt: _Array,
    reduction: Reduction = "mean",
) -> tuple[_Array, _Array]:
    """Trả về (precision, recall) tính theo pixel.

    Nếu mẫu số bằng 0 thì trả 1 khi cả hai mask đều rỗng, còn lại trả 0.
    """
    pred_binary, gt_binary, use_torch = _prepare_pair(pred, gt)
    tp, fp, fn, _ = _metric_counts(
        _counts(pred_binary, gt_binary, use_torch), reduction
    )
    # precision: trong số pixel model tô, bao nhiêu phần đúng.
    # recall: trong số pixel GT, model tô được bao nhiêu phần.
    # both_empty bằng 1 khi cả hai mask đều rỗng (tp + fp + fn = 0), còn lại bằng 0,
    # cộng vào để trường hợp rỗng cả hai được tính là 1 thay vì 0
    both_empty = _ratio(tp * 0, tp + fp + fn, 1.0, use_torch)
    precision = _ratio(tp, tp + fp, 0.0, use_torch) + both_empty
    recall = _ratio(tp, tp + fn, 0.0, use_torch) + both_empty
    return (
        _reduce_metric(precision, reduction, use_torch),
        _reduce_metric(recall, reduction, use_torch),
    )
