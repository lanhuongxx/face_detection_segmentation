# TV4-YEN: Kết quả đánh giá Segmentation trên tập test (YE-3, YE-4, YE-5)

Tài liệu này ghi lại số liệu thật của Yến, cách đo và cách chạy lại, để người khác kiểm tra được. Giao thức đánh giá chung nằm ở [`evaluation_plan.md`](evaluation_plan.md). Cách dùng từng công cụ nằm ở [`TV4-YEN_error_analysis.md`](TV4-YEN_error_analysis.md) và [`TV4-YEN_seg_metrics.md`](TV4-YEN_seg_metrics.md).

## 1. Thiết lập đo

| Mục | Giá trị |
|---|---|
| Dữ liệu | CelebAMask-HQ, 30.000 ảnh có mask da, 29.300 ảnh có nhãn tóc (khớp với notebook của Hương) |
| Chia tập | Đúng cách của Hương: lấy ảnh có nhãn da, xáo bằng `random.Random(42)`, giữ 6.000 ảnh, chia 4.800 train / 600 val / 600 test |
| GT | Mask mặt đã làm sạch theo Hương (hợp 9 nhãn mặt, trừ tóc, bỏ phần mảnh, kernel 31) |
| Pipeline | `FacePipeline` của Hương: RetinaFace `mobile0.25` + U-Net `unet_best.pt` |
| Cấu hình | `configs/pipeline.yaml`: `mask_threshold` 0.5, `conf_threshold` 0.5, crop `margin` 0.25, `size` 256, `min_mask_ratio` 0.30 |
| Phiên bản code | Commit `9f8a902` |
| Trọng số | `models/unet_best.pt`, SHA-256: `0EE16D1AF82876C0BA36C195EAEF7A5F6C1E003E9A1F80787E96DFE21439E584` |
| Môi trường | Python 3.14.5, torch 2.15.0+cpu, chạy trên CPU (không có GPU) |
| Ngày chạy | 22/11/2026 |

Thông tin checkpoint (lấy từ `models/unet_best_info.json` do Hương cung cấp, file này không nằm trên git vì thư mục `models/` bị loại):

```json
{
  "model": "resnet_unet",
  "size": 256,
  "margin": 0.25,
  "max_images": 6000,
  "epochs": 8,
  "batch_size": 8,
  "lr": 0.0003,
  "pretrained": true,
  "clean_labels": true,
  "label_open_kernel": 31,
  "test_dice_raw": 0.9787735939025879,
  "test_iou_raw": 0.9590396285057068,
  "test_dice_clean": 0.9787940531578488,
  "test_iou_clean": 0.9590775791330297,
  "source_checkpoint": "models\\unet_runs\\20261120_080838_558128\\unet_best.pt"
}
```

Điểm test trong file này (Dice 0.9788, IoU 0.9591) là số dev của Hương. Số của Yến ở mục 3 đo độc lập qua pipeline thật và thấp hơn khoảng 0.002 Dice, nên khớp với checkpoint này.

Quy tắc đã giữ: validation dùng để kiểm tra, **tập test chỉ chạy một lần** sau khi chốt cấu hình, và không sửa gì sau khi nhìn số test.

## 2. Hai không gian đo

- **`orig_clean`**: đo trên ảnh gốc 1024, mask dự đoán ghép ngược về đúng cỡ ảnh gốc (evaluation_plan mục 3.1b).
- **`crop256_clean`**: cắt khung vuông 256 quanh mặt như notebook của Hương, nên so được với số của Hương.

## 3. Kết quả trên tập test (600 ảnh)

| Không gian đo | Dice (mean ± std) | IoU (mean ± std) | Pixel P | Pixel R | Dice micro |
|---|---:|---:|---:|---:|---:|
| `crop256_clean` | 0.9769 ± 0.0224 | 0.9557 ± 0.0392 | 0.9758 | 0.9790 | 0.9776 |
| `orig_clean` | 0.9755 | 0.9532 | 0.9747 | 0.9775 | – |
| Số của Hương (notebook, crop 256, test) | 0.9788 | 0.9590 (thô), 0.9591 (sau làm sạch) | – | – | – |

- Dice trung vị của `crop256_clean` là 0.9830.
- Mục tiêu thử nghiệm Dice ≥ 0.85 trong phiếu đăng ký: **đạt**.
- Số của Yến thấp hơn số trong notebook của Hương khoảng 0.002 Dice và 0.003 IoU trên cùng tập test. Số của Yến đo độc lập và đi qua cả pipeline thật (detector, làm sạch mask, elip dự phòng), còn số của Hương là số dev của người train.
- `detect_rate` bằng 1.0 (mọi ảnh thấy ít nhất một mặt). `fallback_rate` là 0.83% (5 trên 600 ảnh dùng mask elip dự phòng). Trung bình pipeline thấy 1.33 mặt mỗi ảnh.
- Tô thừa nhiều hơn bỏ sót ở 308 ảnh, bỏ sót nhiều hơn tô thừa ở 292 ảnh, nên mô hình không lệch hẳn về một phía.

### Số tham khảo trên validation (600 ảnh)

Không dùng để kết luận đạt hay chưa, chỉ để đối chiếu.

| Không gian đo | Dice | IoU | Pixel P | Pixel R | `fallback_rate` |
|---|---:|---:|---:|---:|---:|
| `crop256_clean` | 0.9740 | 0.9510 | 0.9725 | 0.9772 | 1.83% |
| `orig_clean` | 0.9723 | 0.9481 | 0.9710 | 0.9756 | 1.83% |

### Ngưỡng mask

Notebook của Hương chọn ngưỡng 0.45 trên validation, nhưng Dice chỉ hơn ngưỡng 0.5 khoảng 0.0001 (0.9779 so với 0.9778). Vì chênh lệch quá nhỏ nên giữ 0.5 theo `configs/pipeline.yaml`.

## 4. GT crop so với predicted crop (YE-3)

Chạy trên **200 ảnh đầu** của tập test (tập test đã xáo ngẫu nhiên nên 200 ảnh đầu không thiên lệch theo id).

| Cách cắt | N | Miss | Box IoU | Dice | Giảm so với GT crop |
|---|---:|---:|---:|---:|---:|
| `gt_crop` | 200 | 0.000 | 1.000 | 0.9769 | +0.0000 |
| `pred_crop` (RetinaFace thật) | 200 | 0.000 | 0.726 | 0.9755 | +0.0014 |
| `jitter_shift_x10` | 200 | 0.000 | 0.818 | 0.9768 | +0.0000 |
| `jitter_shift_y10` | 200 | 0.000 | 0.836 | 0.9772 | -0.0004 |
| `jitter_shift_20` | 200 | 0.000 | 0.510 | 0.9786 | -0.0017 |
| `jitter_scale_0.8` | 200 | 0.000 | 0.640 | 0.9785 | -0.0016 |
| `jitter_scale_1.2` | 200 | 0.000 | 0.709 | 0.9751 | +0.0018 |

Cách đọc:

- Cắt theo bbox dự đoán hoặc bbox bị làm lệch làm Dice giảm tối đa khoảng 0.002, và không có ảnh nào bị bỏ sót mặt.
- Các giá trị "giảm" âm (jitter 20%, scale 0.8) nằm trong khoảng nhiễu 0.002, không có nghĩa là cắt lệch cho kết quả tốt hơn.
- Box IoU của `pred_crop` là 0.726 vì bbox RetinaFace bao cả trán và tóc, còn bbox GT lấy từ mask mặt đã bỏ tóc. Đây là hai định nghĩa bbox khác nhau, nên IoU không đến 1.0 chưa chắc là do detector sai.

## 5. Ảnh có Dice thấp nhất trên tập test

Sáu ảnh tệ nhất: `13720`, `19925`, `10890`, `09239`, `27065`, `20437`. Ảnh minh họa nằm ở `results/examples_worst.png` (mỗi hàng gồm ảnh gốc, GT, dự đoán, và ảnh lỗi: TP xanh lá, FP đỏ, FN xanh dương).

Nguyên nhân (nhìn 6 ảnh trong `examples_worst.png`):

- **Có thêm một mảng tô thừa không thuộc mặt chính (13720, 10890, 20437).** Dự đoán có thêm một khối tròn đỏ trên tóc hoặc ở góc ảnh. Hình dạng tròn đều cho thấy đây có thể là elip dự phòng của pipeline cho một khuôn mặt thứ hai do detector bắt nhầm. Điều này khớp với việc trung bình pipeline thấy 1.33 mặt mỗi ảnh và `fallback_rate` khác 0 trong `summary.json`, nhưng chưa kiểm chứng trực tiếp bằng cách đọc log từng ảnh.
- **Tóc phủ lên một nửa mặt (09239, 27065).** GT chỉ gồm phần mặt nhìn thấy (hình lưỡi liềm), còn mô hình tô cả vùng da lộ qua tóc nên có mảng FP đỏ bên phía tóc phủ.
- **Vật che khuất (19925).** Hàng rào dây thép chắn trước mặt, mô hình bỏ sót phần trán và hai bên má (FN xanh dương).

Cả 6 ảnh vẫn có Dice từ 0.77 đến 0.86, nên đây là các ca khó chứ không phải lỗi hỏng hoàn toàn.

Chi tiết Dice theo điều kiện ảnh (độ sáng, độ nét, tương phản, tỉ lệ mặt) nằm ở `results/error_by_condition.csv`.

## 6. Hạn chế

- CelebAMask-HQ chỉ có một mặt to, chính diện, nằm giữa khung, và margin 0.25 của pipeline cho dư chỗ. Vì vậy kết quả mục 4 chưa suy ra được cho mặt nhỏ hoặc ảnh nhiều mặt. Kết quả WIDER FACE của Uyên cho thấy RetinaFace bỏ sót khoảng 54% mặt nhỏ.
- GT là mask đã làm sạch theo cách của Hương, không phải nhãn thô của CelebAMask-HQ. Số này so được với số của Hương nhưng không so trực tiếp với các báo cáo khác trên cùng bộ dữ liệu.
- `crop_robustness` chỉ chạy 200 ảnh (không phải 600) vì mỗi ảnh chạy 7 kiểu cắt nên rất lâu.
- Trung bình pipeline thấy 1.33 mặt mỗi ảnh trong khi GT chỉ có một mặt chính. Có thể mặt phía sau cũng bị tô và tính là pixel thừa ở `orig_clean`, nhưng điều này **chưa kiểm chứng**. Chênh giữa `orig_clean` và `crop256_clean` chỉ khoảng 0.0014 Dice.
- Đo trên ảnh tĩnh, chưa đo video.

## 7. Cách chạy lại

Cần có `models/unet_best.pt` (không nằm trên git, Hương gửi riêng) và dữ liệu CelebAMask-HQ. Đặt `$root` là thư mục chứa `CelebA-HQ-img` và `CelebAMask-HQ-mask-anno`.

```powershell
# 1. Test (600 ảnh, một lần duy nhất)
python -m src.evaluation.evaluate_pipeline --data_root $root --split test --out_dir outputs\pipe_test --save_masks

# 2. Phân tích lỗi
python -m src.evaluation.error_analysis --metrics outputs\pipe_test\orig_clean.csv --image_dir $root\CelebA-HQ-img --gt_dir outputs\pipe_test\gt_clean --pred_dir outputs\pipe_test\pred --out_dir outputs\error_test

# 3. GT crop vs predicted crop (200 ảnh đầu)
python -m src.evaluation.crop_robustness --data_root $root --split test --limit 200 --out_dir outputs\crop

# 4. Bảng và hình
python -m src.evaluation.report_tables --run Test=outputs\pipe_test\crop256_clean.csv --crop-summary outputs\crop\crop_summary.csv --out_dir outputs\report
```

Có thể thêm `--detection-json TEN=file.json[@ngưỡng]` vào bước 4 để ghép kết quả detection của Uyên vào cùng bảng.

## 8. File bằng chứng

Các file nằm trong `docs/evaluation/results/`, sinh ra từ các lệnh ở mục 7.

| File | Nội dung | Sinh từ bước |
|---|---|---|
| `summary.json` | Số tổng hợp hai không gian đo trên tập test | 1 |
| `results_tables.md` | Bảng Markdown dùng cho báo cáo | 4 |
| `dice_boxplot.png` | Phân bố Dice từng ảnh | 4 |
| `crop_summary.csv` | Bảng GT crop so với predicted crop | 3 |
| `examples_worst.png` | Ảnh minh họa 6 ca tệ nhất | 2 |
| `error_by_condition.csv` | Dice theo điều kiện ảnh | 2 |
| `worst_cases.csv` | Danh sách ca tệ nhất kèm metric | 2 |

File CSV từng ảnh (`orig_clean.csv`, `crop256_clean.csv`) nằm trong `outputs/` và không đưa lên git vì có thể sinh lại bằng lệnh ở mục 7.
