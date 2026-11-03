"""Tính các metric segmentation nhị phân cho mảng NumPy và tensor PyTorch."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

try:
    import torch
except ImportError:  # Cho phép dùng NumPy mà không bắt buộc cài PyTorch.
    torch = None

# Ba cách tổng hợp: theo từng ảnh, trung bình macro, hoặc gộp pixel kiểu micro.
Reduction = Literal["per_image", "mean", "global"]
_Array = Any


def _is_tensor(value: Any) -> bool:
    """Kiểm tra kiểu Tensor, đồng thời an toàn khi PyTorch chưa được cài."""
    return torch is not None and isinstance(value, torch.Tensor)


def _prepare_pair(pred: _Array, gt: _Array) -> tuple[_Array, _Array, bool]:
    """Kiểm tra shape/device, chuẩn hóa batch và chuyển cả hai mask thành bool."""
    use_torch = _is_tensor(pred) or _is_tensor(gt)

    if use_torch:
        # Nếu một đầu vào là Tensor thì đưa đầu vào còn lại lên cùng device.
        tensors = [value for value in (pred, gt) if _is_tensor(value)]
        device = tensors[0].device
        if any(value.device != device for value in tensors[1:]):
            raise ValueError("pred and gt tensors must be on the same device")
        # detach để metric trong validation không giữ lại computation graph.
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
        # Đưa một ảnh về batch size 1 để phần tính metric dùng chung với batch.
        if shape[0] == 0 or shape[1] == 0:
            raise ValueError("each mask must contain at least one pixel")
        pred_array = pred_array[None, ...]
        gt_array = gt_array[None, ...]
    elif len(shape) == 4 and shape[1] == 1:
        # Bỏ trục channel đơn; mỗi phần tử còn lại có shape (H, W).
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
    """Nhị phân hóa xác suất bằng 0.5 và mask 0/255 bằng 127."""
    if torch is not None and isinstance(mask, torch.Tensor):
        if mask.dtype == torch.bool:
            # Bool đã biểu diễn mask nhị phân nên không cần áp dụng ngưỡng.
            return mask
        if not (
            torch.is_floating_point(mask)
            or mask.dtype
            in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
        ):
            raise ValueError(f"mask dtype must be numeric or bool, got {mask.dtype}")
        # NaN/Inf làm cho kết quả so sánh và metric không có ý nghĩa.
        if not bool(torch.isfinite(mask).all().item()):
            raise ValueError("masks must contain only finite values")
        if torch.is_floating_point(mask) and mask.numel() and bool(
            (mask.min() < 0).item()
        ):
            raise ValueError(
                "float mask has negative values; looks like logits, apply sigmoid first"
            )
        # Dữ liệu trong [0, 1] được hiểu là xác suất; dải lớn hơn dùng ngưỡng 127
        # để nhận diện mask ảnh thường mã hóa foreground là 255.
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
    # Giữ cùng quy tắc ngưỡng với nhánh PyTorch.
    threshold = 0.5 if mask.size == 0 or np.max(mask) <= 1 else 127
    return mask > threshold


def _counts(pred: _Array, gt: _Array, use_torch: bool) -> tuple[_Array, ...]:
    """Đếm TP/FP/FN/TN theo từng ảnh; mỗi mask đã được chuẩn hóa thành bool."""
    if use_torch:
        dims = (1, 2)
        # TP là phần giao; diện tích từng mask giúp suy ra FP và FN.
        tp = (pred & gt).sum(dim=dims, dtype=torch.int64)
        pred_area = pred.sum(dim=dims, dtype=torch.int64)
        gt_area = gt.sum(dim=dims, dtype=torch.int64)
        fp = pred_area - tp
        fn = gt_area - tp
        # TN là các pixel còn lại ngoài TP, FP và FN.
        total = pred.shape[1] * pred.shape[2]
        tn = total - tp - fp - fn
    else:
        dims = (1, 2)
        # Nhánh NumPy dùng cùng công thức; axis (1, 2) là H và W.
        tp = np.count_nonzero(pred & gt, axis=dims)
        pred_area = np.count_nonzero(pred, axis=dims)
        gt_area = np.count_nonzero(gt, axis=dims)
        fp = pred_area - tp
        fn = gt_area - tp
        total = pred.shape[1] * pred.shape[2]
        tn = total - tp - fp - fn
    return tp, fp, fn, tn


def _reduce(value: _Array, reduction: Reduction, use_torch: bool) -> _Array:
    """Tổng hợp mảng giá trị theo từng ảnh, trung bình hoặc phép cộng."""
    if reduction == "per_image":
        return value
    if reduction == "mean":
        # Chuyển count sang float trước khi lấy trung bình.
        return value.to(torch.float64).mean() if use_torch else value.mean()
    if reduction == "global":
        # Chỉ dùng cộng trực tiếp khi caller yêu cầu tổng số đếm.
        return value.sum()
    raise ValueError(
        "reduction must be 'per_image', 'mean', or 'global'; "
        f"got {reduction!r}"
    )


def _reduce_counts(
    counts: tuple[_Array, ...], reduction: Reduction, use_torch: bool
) -> tuple[_Array, ...]:
    """Áp dụng reduction trực tiếp lên từng loại confusion count."""
    return tuple(_reduce(count, reduction, use_torch) for count in counts)


def _metric_counts(
    counts: tuple[_Array, ...], reduction: Reduction
) -> tuple[_Array, ...]:
    """Chọn count đầu vào cho metric, gộp count trước khi tính ở chế độ micro."""
    if reduction == "global":
        # Micro phải tính tỉ lệ từ tổng TP/FP/FN, không lấy tổng các score từng ảnh.
        return tuple(count.sum() for count in counts)
    if reduction not in ("per_image", "mean"):
        raise ValueError(
            "reduction must be 'per_image', 'mean', or 'global'; "
            f"got {reduction!r}"
        )
    return counts


def _reduce_metric(value: _Array, reduction: Reduction, use_torch: bool) -> _Array:
    """Metric đã tính theo ảnh chỉ cần lấy mean khi yêu cầu macro."""
    return _reduce(value, "mean", use_torch) if reduction == "mean" else value


def _ratio(
    numerator: _Array,
    denominator: _Array,
    zero_value: float,
    use_torch: bool,
) -> _Array:
    """Chia an toàn: thay mẫu số 0 bằng giá trị quy ước của từng metric."""
    if use_torch:
        numerator = numerator.to(torch.float64)
        denominator = denominator.to(torch.float64)
        # clamp tránh tạo phép chia cho 0; torch.where chọn giá trị dự phòng.
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
    # Chỉ chia tại vị trí mẫu số dương; các vị trí còn lại giữ zero_value.
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def confusion_counts(
    pred: _Array,
    gt: _Array,
    reduction: Reduction = "per_image",
) -> tuple[_Array, _Array, _Array, _Array]:
    """Trả (TP, FP, FN, TN), mặc định là các count riêng cho từng ảnh."""
    pred_binary, gt_binary, use_torch = _prepare_pair(pred, gt)
    # Khác với metric, confusion count có thể được cộng trực tiếp ở reduction global.
    return _reduce_counts(
        _counts(pred_binary, gt_binary, use_torch), reduction, use_torch
    )


def dice_score(
    pred: _Array,
    gt: _Array,
    reduction: Reduction = "mean",
) -> _Array:
    """Tính Dice = 2TP/(2TP+FP+FN); hai mask rỗng được quy ước bằng 1."""
    pred_binary, gt_binary, use_torch = _prepare_pair(pred, gt)
    # Với global, gộp count trước rồi mới tính Dice micro.
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
    """Tính IoU = TP/(TP+FP+FN); hai mask rỗng được quy ước bằng 1."""
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
    """Trả (pixel precision, pixel recall); mẫu số 0: bằng 1 khi cả hai mask rỗng, ngược lại bằng 0."""
    pred_binary, gt_binary, use_torch = _prepare_pair(pred, gt)
    tp, fp, fn, _ = _metric_counts(
        _counts(pred_binary, gt_binary, use_torch), reduction
    )
    # Precision đo độ chính xác của pixel dự đoán dương; recall đo độ phủ GT.
    both_empty = _ratio(tp * 0, tp + fp + fn, 1.0, use_torch)
    precision = _ratio(tp, tp + fp, 0.0, use_torch) + both_empty
    recall = _ratio(tp, tp + fn, 0.0, use_torch) + both_empty
    return (
        _reduce_metric(precision, reduction, use_torch),
        _reduce_metric(recall, reduction, use_torch),
    )
