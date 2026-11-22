"""Test các hàm metric segmentation (Dice, IoU, precision, recall, confusion counts).

Các test dùng mask nhỏ để tự tính tay được kết quả, rồi so với hàm thật.
"""

import numpy as np
import pytest

# gọi đúng các hàm dùng chung thay vì tự viết lại công thức trong test,
# nếu không thì test sai theo công thức sai thì không phát hiện được
from src.evaluation.seg_metrics import (
    confusion_counts,
    dice_score,
    iou_score,
    pixel_precision_recall,
)


def test_identical_masks_score_one():
    """Dự đoán giống hệt GT thì Dice và IoU phải bằng 1."""
    # giao của hai mask bằng chính diện tích mỗi mask
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    assert dice_score(mask, mask) == pytest.approx(1.0)
    assert iou_score(mask, mask) == pytest.approx(1.0)


def test_non_overlapping_masks_score_zero():
    """Hai vùng foreground không chung pixel nào thì Dice và IoU bằng 0."""
    # pixel dương của hai mask nằm ở hai chỗ khác nhau nên TP = 0
    pred = np.array([[1, 0], [0, 0]])
    gt = np.array([[0, 1], [0, 0]])
    assert dice_score(pred, gt) == pytest.approx(0.0)
    assert iou_score(pred, gt) == pytest.approx(0.0)


def test_four_by_four_overlap_has_hand_calculated_scores():
    """Ví dụ 4x4 đã tính tay: mỗi mask 4 pixel dương, chung nhau 2 pixel."""
    # TP = 2, FP = 2, FN = 2, TN = 16 - 6 = 10
    pred = np.zeros((4, 4), dtype=np.uint8)
    gt = np.zeros((4, 4), dtype=np.uint8)
    pred.flat[:4] = 1
    gt.flat[2:6] = 1
    tp, fp, fn, tn = confusion_counts(pred, gt)
    assert (tp[0], fp[0], fn[0], tn[0]) == (2, 2, 2, 10)
    # IoU = 2 / (2 + 2 + 2) = 1/3, Dice = 4 / (4 + 2 + 2) = 1/2
    assert iou_score(pred, gt) == pytest.approx(1 / 3)
    assert dice_score(pred, gt) == pytest.approx(0.5)
    precision, recall = pixel_precision_recall(pred, gt)
    assert precision == pytest.approx(0.5)
    assert recall == pytest.approx(0.5)


def test_both_empty_masks_score_one():
    """Hai mask cùng rỗng thì tính là khớp hoàn toàn (Dice = IoU = 1)."""
    # đây là quy ước để công thức không phải chia cho 0
    empty = np.zeros((3, 5), dtype=np.uint8)
    assert dice_score(empty, empty) == pytest.approx(1.0)
    assert iou_score(empty, empty) == pytest.approx(1.0)


def test_one_empty_mask_scores_zero():
    """Chỉ một bên rỗng thì Dice và IoU bằng 0."""
    # GT có foreground nhưng dự đoán rỗng nên không có pixel nào chung
    empty = np.zeros((2, 2), dtype=np.uint8)
    full = np.ones((2, 2), dtype=np.uint8)
    assert dice_score(empty, full) == pytest.approx(0.0)
    assert iou_score(empty, full) == pytest.approx(0.0)


def test_precision_recall_both_empty_masks_return_one():
    """Cả hai mask rỗng thì precision và recall đều bằng 1 (theo YE-1)."""
    empty = np.zeros((2, 2), dtype=np.uint8)
    precision, recall = pixel_precision_recall(empty, empty)
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(1.0)


def test_precision_recall_empty_prediction_and_nonempty_gt_return_zero():
    """Dự đoán rỗng nhưng GT có pixel thì precision và recall đều bằng 0 (theo YE-1)."""
    # TP = 0, FP = 0, FN = 1
    pred = np.zeros((2, 2), dtype=np.uint8)
    gt = np.array([[1, 0], [0, 0]], dtype=np.uint8)
    precision, recall = pixel_precision_recall(pred, gt)
    assert precision == pytest.approx(0.0)
    assert recall == pytest.approx(0.0)


def test_precision_recall_nonempty_prediction_and_empty_gt_return_zero():
    """Dự đoán có pixel nhưng GT rỗng thì precision và recall đều bằng 0 (theo YE-1)."""
    # TP = 0, FP = 1, FN = 0
    pred = np.array([[1, 0], [0, 0]], dtype=np.uint8)
    gt = np.zeros((2, 2), dtype=np.uint8)
    precision, recall = pixel_precision_recall(pred, gt)
    assert precision == pytest.approx(0.0)
    assert recall == pytest.approx(0.0)


def test_precision_recall_per_image_handles_empty_and_nonempty_masks():
    """Batch hai ảnh, tính riêng từng ảnh: một ảnh rỗng cả hai, một ảnh bỏ sót."""
    # ảnh 1 rỗng cả hai nên (precision, recall) = (1, 1); ảnh 2 có một FN nên (0, 0)
    pred = np.array(
        [[[[0, 0], [0, 0]]], [[[0, 0], [0, 0]]]], dtype=np.uint8
    )
    gt = np.array(
        [[[[0, 0], [0, 0]]], [[[1, 0], [0, 0]]]], dtype=np.uint8
    )
    precision, recall = pixel_precision_recall(pred, gt, reduction="per_image")
    np.testing.assert_allclose(precision, [1.0, 0.0])
    np.testing.assert_allclose(recall, [1.0, 0.0])


def test_negative_float_mask_raises_value_error():
    """Mask float có giá trị âm phải báo lỗi, không được tự lấy ngưỡng."""
    # giá trị âm gần như chắc chắn là logits chưa qua sigmoid, lấy ngưỡng ra số vô nghĩa
    pred = np.array([[-0.1, 0.8]], dtype=np.float32)
    gt = np.array([[0, 1]], dtype=np.uint8)
    with pytest.raises(
        ValueError,
        match="float mask has negative values; looks like logits, apply sigmoid first",
    ):
        dice_score(pred, gt)


def test_negative_torch_float_mask_raises_value_error():
    """Giống test trên nhưng với torch.Tensor."""
    torch = pytest.importorskip("torch")
    # nhánh torch cũng phải từ chối logits âm trước khi lấy ngưỡng
    pred = torch.tensor([[[[-0.1, 0.8]]]], dtype=torch.float32)
    gt = torch.tensor([[[[0, 1]]]], dtype=torch.uint8)
    with pytest.raises(
        ValueError,
        match="float mask has negative values; looks like logits, apply sigmoid first",
    ):
        dice_score(pred, gt)


def test_255_mask_matches_binary_mask():
    """Mask ảnh 0/255 phải cho kết quả giống mask nhị phân 0/1."""
    pred = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    gt = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    assert dice_score(pred, gt) == pytest.approx(1.0)
    assert iou_score(pred, gt) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "mask",
    [
        np.array([[False, True]], dtype=bool),
        np.array([[0, 1]], dtype=np.uint8),
        np.array([[0.0, 0.9]], dtype=np.float32),
        np.array([[0.0, 255.0]], dtype=np.float32),
    ],
)
def test_bool_uint8_and_float_masks(mask):
    """Bool, số nguyên và số thực đều phải hiểu thành cùng một foreground [False, True]."""
    expected = np.array([[False, True]])
    assert dice_score(mask, expected) == pytest.approx(1.0)
    assert iou_score(mask, expected) == pytest.approx(1.0)


def test_probability_threshold_is_strictly_greater_than_point_five():
    """Ngưỡng xác suất là lớn hơn 0.5 (không phải lớn hơn hoặc bằng)."""
    # đúng 0.5 là nền, chỉ 0.5001 mới là foreground
    pred = np.array([[0.5, 0.5001]], dtype=np.float32)
    gt = np.array([[0, 1]], dtype=np.uint8)
    assert dice_score(pred, gt) == pytest.approx(1.0)


def test_numpy_and_torch_results_match_without_graph():
    """Cùng dữ liệu thì NumPy và torch ra cùng kết quả, và kết quả torch không giữ graph."""
    torch = pytest.importorskip("torch")
    # requires_grad=True giả lập output của model đang train, metric phải tự detach
    pred = torch.tensor([[[[0.0, 0.8], [0.6, 0.0]]]], requires_grad=True)
    gt = np.array([[[[0, 1], [1, 0]]]], dtype=np.uint8)
    tensor_result = dice_score(pred, gt, reduction="per_image")
    numpy_result = dice_score(
        pred.detach().numpy()[0, 0], gt[0, 0], reduction="per_image"
    )
    assert tensor_result.detach().cpu().numpy() == pytest.approx(numpy_result)
    assert tensor_result.requires_grad is False
    assert tensor_result.device == pred.device


def test_torch_batch_reductions_preserve_device_and_detach():
    """Batch torch: kết quả giữ nguyên device của đầu vào và không giữ graph."""
    torch = pytest.importorskip("torch")
    # có GPU thì thử trên cuda, không có thì dùng cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred = torch.tensor(
        [[[[1, 0]]], [[[0, 0]]]], dtype=torch.float32, device=device,
        requires_grad=True,
    )
    gt = torch.tensor(
        [[[[1, 0]]], [[[0, 1]]]], dtype=torch.uint8, device=device
    )
    per_image = dice_score(pred, gt, reduction="per_image")
    global_score = dice_score(pred, gt, reduction="global")
    # ảnh đầu trùng nhau nên Dice = 1; ảnh sau bỏ sót 1 pixel (FN = 1) nên Dice = 0
    assert per_image.detach().cpu().tolist() == pytest.approx([1.0, 0.0])
    assert global_score.item() == pytest.approx(2 / 3)
    assert not per_image.requires_grad
    assert not global_score.requires_grad
    assert per_image.device == device


def test_batch_results_match_individual_image_results():
    """Tính cả batch một lần phải cho kết quả giống tính từng ảnh một."""
    pred = np.array(
        [[[[1, 0], [0, 0]]], [[[0, 1], [1, 0]]]], dtype=np.uint8
    )
    gt = np.array(
        [[[[1, 0], [0, 0]]], [[[1, 0], [0, 0]]]], dtype=np.uint8
    )
    batch = dice_score(pred, gt, reduction="per_image")
    individual = np.array(
        [
            dice_score(pred[index, 0], gt[index, 0])
            for index in range(len(pred))
        ]
    )
    np.testing.assert_allclose(batch, individual)


def test_different_shapes_raise_instead_of_resizing():
    """Hai mask khác shape phải báo lỗi, không được tự resize cho khớp."""
    # tự resize sẽ làm đổi số pixel TP/FP/FN, tức là đổi luôn kết quả metric
    with pytest.raises(ValueError, match="same shape"):
        dice_score(np.zeros((2, 3)), np.zeros((3, 2)))


def test_full_image_masks_score_one():
    """Mask kín cả ảnh ở cả hai bên thì Dice, IoU bằng 1 và chỉ có TP."""
    # 16 pixel đều là foreground nên TP = 16, FP = FN = TN = 0
    full = np.ones((4, 4), dtype=np.uint8)
    assert dice_score(full, full) == pytest.approx(1.0)
    assert iou_score(full, full) == pytest.approx(1.0)
    assert confusion_counts(full, full) == (
        np.array([16]),
        np.array([0]),
        np.array([0]),
        np.array([0]),
    )


def test_macro_dice_differs_from_global_micro_dice():
    """Macro (trung bình các ảnh) và micro (gộp pixel) cho ra hai con số khác nhau."""
    # Ảnh 1: TP = 90, FP = 10, FN = 10 nên Dice = 0.9. Ảnh 2: TP = 0, FN = 10 nên Dice = 0.
    # Macro = (0.9 + 0) / 2 = 0.45. Micro gộp count trước: 2 * 90 / (2 * 90 + 10 + 20) = 180 / 210
    pred = np.zeros((2, 1, 1, 110), dtype=np.uint8)
    gt = np.zeros_like(pred)
    pred[0, 0, 0, :100] = 1
    gt[0, 0, 0, :90] = 1
    gt[0, 0, 0, 100:110] = 1
    gt[1, 0, 0, :10] = 1

    per_image = dice_score(pred, gt, reduction="per_image")
    assert per_image == pytest.approx([0.9, 0.0])
    assert dice_score(pred, gt, reduction="mean") == pytest.approx(0.45)
    assert dice_score(pred, gt, reduction="global") == pytest.approx(180 / 210)
    assert confusion_counts(pred, gt, reduction="global") == (90, 10, 20, 100)
    precision, recall = pixel_precision_recall(pred, gt, reduction="global")
    assert precision == pytest.approx(0.9)
    assert recall == pytest.approx(90 / 110)


def test_global_iou_is_micro_not_mean_of_image_ious():
    """IoU global là micro (tổng TP chia tổng TP + FP + FN), không phải trung bình IoU từng ảnh."""
    # dùng lại dữ liệu của test trước: TP = 90, FP = 10, FN = 20 nên IoU = 90 / 120 = 0.75
    pred = np.zeros((2, 1, 1, 110), dtype=np.uint8)
    gt = np.zeros_like(pred)
    pred[0, 0, 0, :100] = 1
    gt[0, 0, 0, :90] = 1
    gt[0, 0, 0, 100:110] = 1
    gt[1, 0, 0, :10] = 1
    assert iou_score(pred, gt, reduction="global") == pytest.approx(0.75)
