# Bàn giao UY-3 (Đánh giá Detection): RetinaFace vs Haar (TV2-UYEN)

## 1. Tóm tắt

Task này đánh giá hai detector đã làm ở UY-1 (RetinaFace) và UY-2 (Haar Cascade) trên cùng một tập kiểm thử cố định, bằng Precision, Recall, F1 và AP. Task không huấn luyện model; phần tune chỉ áp dụng cho tham số của Haar.

Luồng:

```text
WIDER FACE -> manifest (tune500 / val) -> tune Haar, khóa config -> predictions JSON -> Precision / Recall / F1 / AP
```

- **Tập tune:** 500 ảnh lấy từ WIDER train (seed 42), chỉ dùng để chọn tham số và ngưỡng score cho Haar.
- **Tập test:** toàn bộ WIDER val (3226 ảnh), chỉ chạy sau khi đã khóa cấu hình.
- **Chỉ số:** IoU >= 0.5 mới tính là khớp; báo cáo cả toàn tập và theo nhóm kích thước mặt (nhỏ / vừa / lớn).

Người phụ trách task: Uyên. Yến phụ trách evaluation plan, Huy phụ trách dữ liệu. Code và test đã xong; **đã chạy cả Haar và RetinaFace trên WIDER val và điền bảng ở mục 6** (mục Kiểm thử và môi trường còn thiếu); còn chờ review.

## 2. Vì sao cần làm

UY-1 và UY-2 mới chỉ tạo ra bbox, chưa có con số nào cho biết detector nào tốt hơn. Cần một phép đo chung để nhóm chọn detector cho pipeline và để báo cáo có số liệu, thay vì chỉ so bằng mắt trên vài ảnh.

Để phép so sánh công bằng, hai detector chạy trên đúng cùng manifest, cùng split và cùng cách ghép box. Haar được tune trên tập train riêng rồi khóa cấu hình trước khi chạy val, nên val không bị dùng để chỉnh tham số (tránh rò rỉ dữ liệu test).

## 3. File và vị trí trong repo

- [`src/detection/wider_data.py`](../../src/detection/wider_data.py): đọc nhãn WIDER FACE, tạo manifest (WIDER val = test, 500 ảnh train = tune).
- [`src/detection/detect_metrics.py`](../../src/detection/detect_metrics.py): ghép box, tính Precision / Recall / F1 / AP, chia nhóm kích thước, CLI tính metric từ file predictions.
- [`src/detection/tune_haar.py`](../../src/detection/tune_haar.py): tune Haar trên 500 ảnh train, khóa cấu hình vào `configs/detection_locked.yaml`.
- [`src/detection/run_detect.py`](../../src/detection/run_detect.py): chạy detector trên manifest, ghi predictions JSON (đã có từ UY-2, dùng lại).
- [`tests/test_detect_metrics.py`](../../tests/test_detect_metrics.py): kiểm thử metric bằng ví dụ tính tay.
- `.gitignore`: loại thêm thư mục `data/wider_face/` để không commit nhầm dữ liệu nặng.

### Đã làm

- Đọc nhãn WIDER FACE (`x y w h blur expression illumination invalid occlusion pose`); đổi box sang `[x1, y1, x2, y2]` với `x2 = x + w`; bỏ qua dòng giữ chỗ toàn số 0 của ảnh không có mặt; báo lỗi nếu một ảnh xuất hiện hai lần trong file nhãn.
- Tạo hai manifest: `manifest_wider_val.csv` (split `test`) và `manifest_wider_tune500.csv` (split `tune`, 500 ảnh train chọn bằng seed 42).
- Viết bộ đánh giá: IoU, ghép box theo score giảm dần, loại mặt `invalid`, tính P/R/F1 tại ngưỡng score cố định và AP (all-point interpolation) trên toàn tập.
- Chia nhóm theo chiều cao box: nhỏ < 32 px, vừa 32 đến 96 px, lớn >= 96 px. GT và prediction ngoài nhóm đang xét được bỏ qua, không bị tính FP.
- Tune Haar bằng lưới 48 cấu hình (mục 5); mỗi cấu hình chọn ngưỡng score cho F1 cao nhất, rồi lấy cấu hình có F1 cao nhất và ghi vào `configs/detection_locked.yaml`.
- Báo lỗi nếu thiếu prediction cho ảnh nào trong tập đánh giá (tránh recall bị ảo); có `--allow-partial` cho lúc chạy thử.
- Thêm test cho IoU, AP, quy tắc ghép, mặt invalid, đọc nhãn, nhóm kích thước và luồng đánh giá đầu-cuối.

## 4. Lý thuyết cần biết

**IoU** (Intersection over Union) là diện tích giao chia diện tích hợp của hai box. Ví dụ hai box 10x10 chồng nhau 5x10: giao 50, hợp 150, IoU = 1/3. Prediction chỉ được coi là khớp GT khi IoU >= 0.5.

**TP / FP / FN:** với mỗi ảnh, prediction được xếp theo score giảm dần. Mỗi prediction ghép với GT hợp lệ chưa dùng có IoU cao nhất; nếu IoU >= 0.5 thì là TP. Không ghép được là FP. GT hợp lệ không được ghép là FN. Mỗi GT chỉ ghép tối đa một lần, nên hai box cùng trúng một mặt thì box thứ hai là FP.

**Mặt `invalid`:** WIDER đánh dấu một số mặt `invalid=1` (quá nhỏ hoặc không rõ). Các mặt này bị loại khỏi GT; prediction trùng mặt invalid được bỏ qua, không tính FP.

**Precision, Recall, F1:** Precision = TP / (TP + FP); Recall = TP / (TP + FN); F1 = 2PR / (P + R). Quy ước: nếu không có prediction và không có GT thì cả ba bằng 1; nếu có GT mà không đoán được gì thì cả ba bằng 0.

**AP (Average Precision):** sắp mọi prediction của cả tập theo score giảm dần, cộng dồn TP/FP để có precision và recall ở từng mức score, lấy precision lớn nhất bên phải mỗi mức recall (envelope) rồi tính diện tích dưới đường precision-recall. Ví dụ có 4 GT, prediction theo thứ tự TP, TP, FP, TP, FP:

| Prediction | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| Kết quả | TP | TP | FP | TP | FP |
| Precision | 1 | 1 | 2/3 | 3/4 | 3/5 |
| Recall | 1/4 | 2/4 | 2/4 | 3/4 | 3/4 |

Sau khi lấy precision lớn nhất bên phải: AP = 0.25 x 1 + 0.25 x 1 + 0.25 x 0.75 = **0.6875**. Ví dụ này có trong `test_ap_example_from_plan`.

**Vì sao RetinaFace chạy với `conf_thres=0.01`:** AP cần cả đường precision-recall, nên phải giữ gần hết box. Nếu cắt ở 0.5 thì đường cong bị cụt và AP thấp hơn thực tế. Sau đó lọc `score >= 0.5` để tính P/R/F1. Chạy riêng với 0.5 cho cùng kết quả P/R/F1 vì NMS giữ box score cao trước.

**Score của Haar:** Haar không có xác suất, nên dùng sigmoid của `levelWeight` (xem UY-2). Score này chỉ dùng để xếp hạng, vì vậy **AP của Haar chỉ để tham khảo, không so trực tiếp với AP của RetinaFace**. Haar so với RetinaFace nên dựa vào P/R/F1 tại ngưỡng đã khóa.

**Giới hạn:** kết quả chỉ so sánh được trong nhóm, không so với bảng xếp hạng WIDER chính thức, vì nhóm dùng nhóm kích thước theo chiều cao box thay vì Easy/Medium/Hard của WIDER và có thể khác cách xử lý chi tiết.

## 5. Chạy lệnh 
### Cấu hình Haar sau khi tune

Lưới tham số, tổng 3 x 4 x 2 x 2 = 48 cấu hình:

| Tham số | Giá trị thử |
|---|---|
| `scale_factor` | 1.05, 1.1, 1.2 |
| `min_neighbors` | 2, 3, 5, 8 |
| `min_size` | [24, 24], [30, 30] |
| `equalize_hist` | false, true |

Mỗi cấu hình được đánh giá trên 500 ảnh tune và chọn ngưỡng score cho F1 cao nhất. Cấu hình có F1 cao nhất được ghi vào `configs/detection_locked.yaml` cùng `score_thr` của Haar; RetinaFace cố định `score_thr = 0.5`. File này **phải tạo trước khi chạy WIDER val và không sửa tay sau đó**.

### Lệnh chạy (theo thứ tự)

```powershell
python -m src.detection.wider_data --root data\wider_face
python -m src.detection.tune_haar --manifest data\wider_face\manifest_wider_tune500.csv --gt data\wider_face\wider_face_split\wider_face_train_bbx_gt.txt
python -m src.detection.run_detect --detector haar --config configs\detection_locked.yaml --manifest data\wider_face\manifest_wider_val.csv --out outputs\detection\haar_val.json
python -m src.detection.run_detect --detector retinaface --conf-thres 0.01 --manifest data\wider_face\manifest_wider_val.csv --out outputs\detection\retinaface_val.json
python -m src.detection.detect_metrics --detector haar --pred outputs\detection\haar_val.json --gt data\wider_face\wider_face_split\wider_face_val_bbx_gt.txt --out outputs\detection\haar_metrics.json
python -m src.detection.detect_metrics --detector retinaface --pred outputs\detection\retinaface_val.json --gt data\wider_face\wider_face_split\wider_face_val_bbx_gt.txt --out outputs\detection\retinaface_metrics.json
```

`--detector` trong `detect_metrics` đọc `score_thr` từ `configs/detection_locked.yaml`, nên phải chạy `tune_haar` trước. Muốn không phụ thuộc file này thì truyền thẳng `--score-thr 0.5`.

RetinaFace trên CPU chạy khá lâu. Nên thử trước trên một manifest nhỏ (vài chục ảnh) rồi mới chạy đủ 3226 ảnh; khi tính metric cho tập nhỏ thì thêm `--allow-partial`, nếu không `detect_metrics` sẽ báo thiếu prediction. Predictions và kết quả nằm trong `outputs/`, đã được `.gitignore` loại khỏi Git.

Chạy test:

```powershell
python -m pytest -q
```

### Cách đọc kết quả

`detect_metrics` in một bảng, mỗi dòng là một nhóm (`all`, `small`, `medium`, `large`) với các cột `GT`, `TP`, `FP`, `FN`, `P`, `R`, `F1`, `AP`. Dòng `all` là kết quả toàn tập; ba dòng còn lại chỉ tính mặt trong nhóm kích thước tương ứng. Với `--out`, kết quả được ghi ra JSON (không kèm dữ liệu đường cong).

## 6. Kết quả và số liệu

### Kiểm thử

- `test_detect_metrics.py` có 10 test, đều pass. Chạy cùng `test_haar_baseline.py` với OpenCV 4.x: `29 passed, 1 skipped` (không gồm `test_retinaface.py` vì cần PyTorch và FaceXLib). Test bị skip cần ảnh mặt frontal thật (biến môi trường `UY2_FACE_IMAGE`).
- Kiểm thử toàn bộ repo: `python -m pytest -q`: `57 passed, 1 skipped in 3.99s`. Test bị skip là test cần ảnh mặt frontal thật (biến môi trường `UY2_FACE_IMAGE`, xem ở trên)..
- Môi trường: Python 3.14.5, OpenCV 4.14.0, NumPy 2.5.4, PyTorch 2.15.0+cpu; phần cứng: CPU Intel Core i7-1255U (thế hệ 12), không dùng GPU (CUDA không khả dụng).
- Không dùng seed cho inference; seed 42 chỉ dùng để chọn 500 ảnh tune.

### Kết quả trên WIDER val

> Số liệu của cả hai detector lấy từ lần chạy đủ 3226 ảnh WIDER val (Haar với config đã khóa, RetinaFace với `conf_thres=0.01` rồi lọc `score >= 0.5`).

Toàn tập (`all`), IoU 0.5:

| Detector | Ngưỡng score | Precision | Recall | F1 | AP |
|---|---:|---:|---:|---:|---:|
| Haar (config đã khóa) | 0.693463 | 0.5026 | 0.1787 | 0.2637 | 0.1581 (chỉ tham khảo) |
| RetinaFace | 0.5 | 0.9128 | 0.6095 | 0.7309 | 0.7253 |

Theo nhóm kích thước:

| Nhóm | Detector | GT | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| Nhỏ (< 32 px) | Haar | 26450 | 0.5900 | 0.0510 | 0.0940 |
| Nhỏ (< 32 px) | RetinaFace | 26450 | 0.8811 | 0.4592 | 0.6038 |
| Vừa (32 đến 96 px) | Haar | 9765 | 0.4760 | 0.4097 | 0.4404 |
| Vừa (32 đến 96 px) | RetinaFace | 9765 | 0.9410 | 0.9096 | 0.9250 |
| Lớn (>= 96 px) | Haar | 2897 | 0.5100 | 0.5661 | 0.5366 |
| Lớn (>= 96 px) | RetinaFace | 2897 | 0.9720 | 0.9710 | 0.9715 |

Cấu hình Haar được chọn khi tune (500 ảnh train, đã khóa vào `configs/detection_locked.yaml`): `scale_factor=1.1`, `min_neighbors=2`, `min_size=[24, 24]`, `equalize_hist=false`, `score_thr=0.693463`. Trên tập tune: F1 = 0.2796, Precision = 0.5713, Recall = 0.1851.

Nhận xét về kết quả tune: Haar chỉ đạt F1 khoảng 0.28 ở cấu hình tốt nhất, recall thấp (0.19) vì WIDER có nhiều mặt nhỏ, nghiêng, bị che mà Haar bỏ sót. Xu hướng trong lưới 48 cấu hình: `min_size=[24, 24]` luôn tốt hơn `[30, 30]`; `equalize_hist=true` luôn kém hơn `false`; `min_neighbors` thấp (2 đến 3) cho F1 cao hơn 5 và 8; `scale_factor` 1.05 và 1.1 tốt hơn 1.2 (F1 tune cao nhất của `scale_factor=1.05` là 0.2762, của 1.1 là 0.2796, của 1.2 là 0.2578). Đây là số trên tập tune, không phải kết quả đánh giá cuối.

### Nhận xét

Haar trên WIDER val (toàn tập 39112 GT): TP = 6991, FP = 6919, FN = 32121. Precision 0.5026 nhưng Recall chỉ 0.1787, nên F1 = 0.2637 (gần với F1 0.2796 trên tập tune, không có dấu hiệu quá khớp).

Theo kích thước, Haar yếu nhất ở mặt nhỏ: nhóm nhỏ chiếm 26450/39112 GT (khoảng 68%) mà Recall chỉ 0.0510 (F1 0.0940). Recall tăng dần theo kích thước: 0.4097 ở nhóm vừa (F1 0.4404) và 0.5661 ở nhóm lớn (F1 0.5366). Vì phần lớn GT là mặt nhỏ nên Recall toàn tập thấp. Ngay cả ở nhóm lớn Haar vẫn bỏ sót khoảng 43% mặt và Precision chỉ khoảng 0.5.

RetinaFace (toàn tập): TP = 23839, FP = 2277, FN = 15273; Precision 0.9128, Recall 0.6095, F1 = 0.7309, AP = 0.7253.

**So sánh:** RetinaFace tốt hơn Haar ở mọi nhóm và mọi chỉ số P/R/F1. Toàn tập F1 là 0.7309 so với 0.2637, Precision 0.9128 so với 0.5026, Recall 0.6095 so với 0.1787. (AP của Haar chỉ để tham khảo nên không đặt cạnh AP của RetinaFace.)

| Nhóm | F1 Haar | F1 RetinaFace | Chênh | Recall Haar | Recall RetinaFace |
|---|---:|---:|---:|---:|---:|
| Nhỏ | 0.0940 | 0.6038 | +0.5098 | 0.0510 | 0.4592 |
| Vừa | 0.4404 | 0.9250 | +0.4846 | 0.4097 | 0.9096 |
| Lớn | 0.5366 | 0.9715 | +0.4349 | 0.5661 | 0.9710 |

Chênh F1 lớn nhất ở nhóm nhỏ (+0.51), nhưng nhóm vừa cũng gần như vậy (+0.48); nhóm lớn chênh ít nhất (+0.43) vì Haar làm tốt nhất ở đây. Ở nhóm nhỏ, Haar bắt được 5.1% mặt, RetinaFace bắt được 45.9%. Ở nhóm vừa và lớn RetinaFace đạt Recall khoảng 0.91 đến 0.97 trong khi Haar chỉ 0.41 đến 0.57.

**Hạn chế còn lại của RetinaFace:** mặt nhỏ vẫn là điểm yếu chính. Nhóm nhỏ chiếm khoảng 68% GT và RetinaFace bỏ sót 14304/26450 mặt (khoảng 54%), nên Recall toàn tập chỉ 0.6095 dù Precision cao.

**Kết luận:** nên chọn RetinaFace cho pipeline. Phần "Haar sai ở loại ảnh nào" (ví dụ ảnh đông người, mặt nghiêng, che khuất) cần xem ảnh dự đoán cụ thể; bảng số hiện tại chưa cho biết, nên chưa kết luận.

## 7. Checklist Definition of Done

- [x] Đọc được nhãn WIDER FACE, tạo manifest val (split `test`) và tune500 (split `tune`, seed 42).
- [x] Metric P/R/F1/AP theo quy tắc ghép đã mô tả, loại mặt invalid, chia nhóm kích thước.
- [x] Tune Haar trên tập train riêng và khóa cấu hình vào `configs/detection_locked.yaml`.
- [x] Pytest cho IoU, AP (ví dụ tính tay 0.6875), quy tắc ghép, đọc nhãn, nhóm kích thước và luồng đầu-cuối.
- [x] Chạy full test suite và điền kết quả vào mục 6.
- [x] Chạy WIDER val cho cả hai detector và điền bảng ở mục 6.
- [ ] Yến xác nhận cách tính khớp với evaluation plan.

- [ ] Một thành viên khác review/approve PR; cập nhật trạng thái task và backlog.