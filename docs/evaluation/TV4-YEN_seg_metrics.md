# Bàn giao YE-2 (Task Segmentation): IoU/Dice evaluator và unit test (TV4-YEN)

## 1. Tóm tắt

Task này xây dựng metric nhị phân cho segmentation và công cụ đánh giá mask theo từng ảnh. Module hỗ trợ NumPy và PyTorch để có thể dùng chung trong vòng validation hoặc khi đánh giá thư mục kết quả.

Luồng đánh giá:

```text
Mask dự đoán + ground truth -> nhị phân hóa -> TP/FP/FN/TN -> Dice, IoU, precision, recall -> CSV và thống kê
```

Module không resize mask. Nếu shape prediction và ground truth khác nhau, evaluator báo lỗi để tránh âm thầm làm sai kết quả.

## 2. Vì sao cần làm

Segmentation cần metric thống nhất để theo dõi chất lượng trong lúc train và so sánh kết quả validation/test. Dice và IoU đo mức giao giữa mask dự đoán với ground truth; pixel precision và recall cho biết tỷ lệ dự đoán dương đúng và tỷ lệ pixel dương thật được tìm thấy.

Macro tính metric riêng cho từng ảnh rồi lấy trung bình, tránh để ảnh lớn lấn át ảnh nhỏ. Micro cộng TP/FP/FN của toàn bộ pixel rồi mới tính metric; đây là số đo bổ sung phản ánh kết quả trên toàn tập.

## 3. File và vị trí trong repo

Các file thuộc phạm vi YE-2:

- `src/evaluation/seg_metrics.py`: confusion counts, Dice, IoU, pixel precision/recall.
- `src/evaluation/evaluate.py`: CLI ghép cặp mask, ghi CSV và in thống kê.
- `tests/test_seg_metrics.py`: unit test metric và các ví dụ tính tay.
- `tests/test_evaluate.py`: test ghép cặp, báo lỗi thiếu mask, CSV và manifest.
- `requirements.txt`: NumPy và Pillow để chạy evaluator.
- `docs/evaluation/YE-2_seg_metrics.md`: tài liệu bàn giao task.

### Đã làm

- Thêm `confusion_counts(pred, gt)` trả `(TP, FP, FN, TN)`.
- Thêm `dice_score`, `iou_score`, `pixel_precision_recall`.
- Hỗ trợ đầu vào NumPy hoặc PyTorch với shape `(H, W)` hoặc `(N, 1, H, W)`. Có thể truyền Tensor trên GPU; hàm detach tensor đầu vào để metric không giữ graph và trả tensor metric trên cùng device.
- Chuẩn hóa mask xác suất bằng ngưỡng `> 0.5`; mask 0/255 bằng `> 127`. Hỗ trợ bool, uint8 và float.
- Báo `ValueError` khi hai mask lệch shape, mask sai shape hoặc chứa giá trị không hữu hạn; không tự resize.
- Quy ước hai mask rỗng có Dice/IoU bằng `1.0`; một mask rỗng có Dice/IoU bằng `0.0`. Cả hai mask rỗng thì precision/recall bằng `1`; ngược lại mẫu số `0` thì bằng `0` (theo YE-1 mục 3.1).
- Bổ sung reduction `per_image`, `mean` (macro) và `global` (micro). `confusion_counts` mặc định trả số đếm theo ảnh; các metric mặc định trả macro.
- Thêm CLI nhận thư mục prediction/ground truth hoặc CSV manifest. Thiếu cặp, trùng tên hoặc thiếu file sẽ báo lỗi.
- CSV chứa `image_id, dice, iou, precision, recall, area_gt, area_pred`; CLI in mean, population std, median, min, max và metric micro.

## 4. Lý thuyết cần biết

Với TP là pixel mặt được dự đoán đúng, FP là pixel nền bị dự đoán thành mặt, FN là pixel mặt bị bỏ sót:

```text
IoU       = TP / (TP + FP + FN)
Dice      = 2TP / (2TP + FP + FN)
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
```

Ví dụ mask `4 x 4` có 4 pixel mặt ở ground truth và 4 pixel mặt ở prediction, trong đó chồng lấp 2 pixel: TP=2, FP=2, FN=2. Khi đó IoU=`1/3`, Dice=`0.5`, precision=`0.5`, recall=`0.5`.

Các reduction:

- `per_image`: trả metric riêng cho từng ảnh.
- `mean`: tính metric mỗi ảnh rồi lấy trung bình macro.
- `global`: cộng confusion counts trước khi tính metric micro.

Ví dụ macro/micro: ảnh 1 có TP=90, FP=10, FN=10; ảnh 2 có GT 10 pixel và prediction rỗng. Dice từng ảnh là `0.9` và `0.0`, nên macro Dice=`0.45`. Micro Dice=`180/210`, xấp xỉ `0.857`.

## 5. Cách dùng và lệnh chạy

### Gọi metric trong train.py

`pred_mask` nên là xác suất sau sigmoid (hoặc mask nhị phân); `masks` là ground truth có cùng shape:

```python
import torch

from src.evaluation.seg_metrics import dice_score, iou_score
pred_mask = torch.sigmoid(logits)
val_dice = dice_score(pred_mask, masks, reduction="mean")
val_iou = iou_score(pred_mask, masks, reduction="mean")
print(f"val Dice={val_dice.item():.4f}, IoU={val_iou.item():.4f}")
```

Có thể lấy metric riêng từng ảnh bằng `reduction="per_image"` hoặc metric micro bằng `reduction="global"`. PyTorch là tùy chọn cho metric; để dùng Torch/GPU cần cài PyTorch tương ứng với máy.

Lưu ý:
- `pred_mask` phải là xác suất sau sigmoid; nếu truyền logits có giá trị âm, code sẽ báo lỗi.
- Mask theo batch phải có shape `(N, 1, H, W)`.

### Chạy evaluator trên thư mục

Ghép cặp bằng tên file tương đối trong thư mục, bỏ phần mở rộng. Hỗ trợ ảnh `.bmp`, `.jpeg`, `.jpg`, `.png`, `.tif`, `.tiff` và file `.npy`; mỗi file phải chứa một mask 2D.

```powershell
python -m src.evaluation.evaluate --pred_dir predictions --gt_dir ground_truth --out outputs/eval/seg_metrics.csv
```

Nếu không truyền `--out`, kết quả được ghi mặc định vào `outputs/eval/seg_metrics.csv`.

### Chạy bằng manifest

Manifest là CSV có ba cột `image_id,pred_path,gt_path`. Đường dẫn tương đối được tính từ thư mục chứa manifest:

```csv
image_id,pred_path,gt_path
0001,predictions/0001.png,ground_truth/0001.png
```

Chạy lệnh:

```powershell
python -m src.evaluation.evaluate --manifest eval_manifest.csv --out outputs/eval/seg_metrics.csv
```

### Chạy test

```powershell
python -m pytest -q
```

## 6. Kết quả và số liệu

- `python -m pytest -q`: số test pass sẽ được cập nhật sau khi chạy kiểm tra.
- Test gồm các đáp án tính tay: mask giống hệt, không giao nhau, ví dụ `4 x 4`, hai mask rỗng, một mask rỗng, mask 0/255 với 0/1, bool/uint8/float, shape lệch, mask phủ toàn ảnh, macro và micro.
- Test Torch xác nhận kết quả khớp NumPy, tensor không giữ graph và metric batch giữ device. Test PyTorch chưa chạy trên GPU.
- Smoke test `python -m src.evaluation.evaluate` đã ghi CSV đúng, in thống kê theo từng ảnh và micro. Test riêng xác nhận ghép cặp theo tên giữa phần mở rộng khác nhau và đường dẫn manifest tương đối.
- Benchmark metric: **1.000 mask, mỗi mask `512 x 512`, NumPy CPU**. Đo cả bốn API (`confusion_counts`, Dice, IoU, precision/recall) trên cùng bộ batch, gồm 20 batch x 50 ảnh, sau một lượt warm-up cho từng API. Tổng thời gian gọi metric là **9.120 giây**, tương đương **9.12 ms/ảnh** cho cả bốn API hoặc khoảng **109.6 ảnh/giây**. Sinh dữ liệu nằm ngoài khoảng bấm giờ; đây là benchmark tổng hợp, không phải thời gian inference của model.
- Môi trường đo: Python `3.14.0`, NumPy `2.5.4`, PyTorch `2.15.0+cpu`, Windows; không có CUDA.
