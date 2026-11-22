# Bàn giao Yến (TV4): đánh giá U-Net, phân tích lỗi, GT crop vs predicted crop, bảng kết quả

## 1. Tóm tắt

Phần việc của Yến theo `evaluation_plan.md`: đo IoU/Dice, phân tích lỗi theo điều kiện ảnh, so sánh GT crop với predicted crop, và bảng/hình cho báo cáo. Đo trên đúng tập val/test của Hương (80/10/10, seed 42), chạy `FacePipeline` thật của Hương.

Trạng thái: code và test xong (chạy bằng engine giả). **Chưa có số thật**: cần `models/unet_best.pt` của Hương và dữ liệu CelebAMask-HQ.

## 2. File và cách chạy (từ thư mục gốc repo, dữ liệu ở `data/raw/CelebAMask-HQ`)

| File | Việc |
|---|---|
| `seg_metrics.py`, `evaluate.py` | IoU/Dice/P/R theo ảnh (có sửa lỗi PNG 16-bit, mục 4) |
| `huong_protocol.py` | Chia tập và mask mặt đúng như notebook của Hương |
| `pipeline_adapter.py` | Gọi `FacePipeline` của Hương, trả mask cỡ ảnh gốc |
| `evaluate_pipeline.py` | Đo pipeline trên val/test, ghi CSV và mask |
| `error_analysis.py` | Ví dụ TP/FP/FN, Dice theo độ sáng/độ nét/tương phản/cỡ mặt |
| `crop_robustness.py` | GT crop, predicted crop, crop bị lệch |
| `report_tables.py` | Bảng Markdown và hình từ log (đọc cả kết quả của Uyên) |

```powershell
# 1. Validation (chọn checkpoint/ngưỡng)
python -m src.evaluation.evaluate_pipeline --data_root data\raw\CelebAMask-HQ --split val --weights models\unet_best.pt --out_dir outputs\pipe_val --save_masks
# 2. Test: chỉ chạy đúng MỘT lần, sau khi đã khóa checkpoint và ngưỡng
python -m src.evaluation.evaluate_pipeline --data_root data\raw\CelebAMask-HQ --split test --weights models\unet_best.pt --out_dir outputs\pipe_test --save_masks
# 3. Phân tích lỗi trên test
python -m src.evaluation.error_analysis --metrics outputs\pipe_test\orig_clean.csv --image_dir data\raw\CelebAMask-HQ\CelebA-HQ-img --gt_dir outputs\pipe_test\gt_clean --pred_dir outputs\pipe_test\pred
# 4. GT crop vs predicted crop
python -m src.evaluation.crop_robustness --data_root data\raw\CelebAMask-HQ --split test --weights models\unet_best.pt --out_dir outputs\crop
# 5. Bảng và hình (--detection-json nhận file của detect_metrics; file không có ngưỡng thì thêm @0.693)
python -m src.evaluation.report_tables --run Test=outputs\pipe_test\crop256_clean.csv --detection-json retinaface=outputs\detection\retinaface_metrics.json --detection-json haar=outputs\detection\haar_metrics.json --crop-summary outputs\crop\crop_summary.csv
```

`evaluate_pipeline` ghi hai CSV: `orig_clean` (ảnh gốc 1024, theo evaluation_plan 3.1b) và `crop256_clean` (crop 256 như notebook, **so được với Dice 0.9788 / IoU 0.9590 của Hương**). GT là mask Hương đã làm sạch (trừ tóc, bỏ tua mảnh), không phải nhãn thô.

Chia tập của Hương: ảnh có nhãn da, sắp theo id, `random.Random(42).shuffle`, lấy 6.000 ảnh, train/val/test = 4.800/600/600 (test = `[-600:]`). Để Hương đối chiếu: `huong_split(ids_with_skin(mask_dir))["test"][:10]` phải bằng `[s[0] for s in test_samples[:10]]` trong notebook.

## 3. Quy tắc

- Chỉ số điều kiện ảnh: độ sáng, độ nét (phương sai Laplacian), tương phản, tỉ lệ diện tích mặt. Mỗi điều kiện chia low/mid/high theo tertile của chính tập đánh giá; ngưỡng được ghi trong `error_by_condition.csv`.
- Metric crop tính trong không gian ảnh gốc: resize xác suất, ngưỡng 0.5, dán vào ảnh trống tại vị trí crop.
- Không phát hiện được mặt (predicted crop) = mask rỗng = Dice 0, được báo riêng trong cột `miss_rate`.
- Micro Dice/IoU trong bảng tổng hợp suy ra từ CSV theo ảnh qua `TP = Dice * (area_gt + area_pred) / 2`, có test đối chiếu với đếm trực tiếp.

## 4. Đã rà code cũ (YE-1, seg_metrics, evaluate) bằng cách chạy thật

| Kiểm tra | Kết quả |
|---|---|
| Test cũ + mới (có cài PyTorch để chạy cả nhánh Tensor) | 48 passed |
| Đối chiếu `dice_score`, `iou_score` với công thức tính tay | 300 cặp mask ngẫu nhiên x 4 kiểu dữ liệu (bool, uint8 0/255, Tensor bool, Tensor float): **0 sai lệch**, kể cả mask rỗng |
| Quy ước mask rỗng (cả hai rỗng = 1, một bên rỗng = 0) | Đúng như `evaluation_plan.md` 3.1 |
| Xác suất đúng bằng 0.5 | Tính là nền (`> 0.5`), khớp tài liệu |

**Lỗi tìm thấy và đã sửa (`src/evaluation/evaluate.py`, hàm `_load_mask`):** PNG 16-bit (ví dụ bản đồ xác suất lưu 0..65535) bị Pillow `convert("L")` cắt về 255, nên mọi pixel thành 255 và Dice sai mà không có cảnh báo. Ví dụ tái hiện: mask GT nửa trên, dự đoán xác suất 0.9 ở nửa trên và 0.1 ở nửa dưới lưu PNG 16-bit: Dice ra **0.667** thay vì **1.0**. Đã sửa: đọc 16-bit theo toàn dải 0..65535 rồi chia 257 về 0..255; mode `F` giữ nguyên xác suất. PNG 8-bit và `.npy` không đổi. Có test `tests/test_evaluate_16bit.py`. Quy ước cho Hương: lưu xác suất bằng PNG 8-bit, `.npy` hoặc PNG 16-bit toàn dải, đều được.

Đã chạy thử `evaluate_pipeline.py` và `crop_robustness.py` bằng engine giả (có test `test_cli_smoke.py`) và RetinaFace thật trên ảnh mẫu; số Dice của các lần thử không có ý nghĩa vì U-Net giả.

## 5. Giới hạn và cảnh báo

- GT crop lấy từ bbox của mask GT, còn RetinaFace cho bbox ôm khuôn mặt chặt hơn (thường không gồm tóc/tai). Hai bbox vốn khác nhau nên cột `box_iou` của predicted crop có thể thấp dù detection đúng; đọc kết quả cùng `box_iou`.
- Các cách lệch trong `crop_robustness.py` là mô phỏng, không phải lỗi detection thật.
- Chỉ test trên dữ liệu giả lập và một ảnh mẫu. Chưa chạy trên CelebAMask-HQ và checkpoint thật.
- **Mask của Huy gộp cả `ear_r`**: trong CelebAMask-HQ nhãn này là *khuyên tai* (tai phải là `r_ear`). Overlay 30 ảnh cho thấy khuyên tai bị tô đỏ (ví dụ ảnh 1819, 21024, 900, 3393). Mục 3.4 của evaluation_plan đang đề xuất không gộp tai/kính. Cần Huy và Hương chốt định nghĩa trước khi train, vì đổi sau đòi hỏi tạo lại mask, train lại và đánh giá lại.

## 6. Việc cần cả nhóm chốt

1. **Ngưỡng:** Hương dùng 0.45 (`configs/pipeline.yaml`), plan ghi 0.5. Chốt trên val trước khi chạy test.
2. **Tập test chung:** Tuấn dùng đúng cách chia và đúng 9 nhãn mặt của Hương (không tai, không `ear_r`), nếu không test có thể lẫn ảnh train của Hương.
3. **Q2 (so sánh loss):** nếu Hương chỉ train một loss thì không có bảng BCE/Dice/BCE+Dice; hỏi Hương đã train loss nào.
4. Cần `models/unet_best.pt` để chạy thật.
