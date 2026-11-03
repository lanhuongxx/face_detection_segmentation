# YE-1 – Giao thức đánh giá

## 1. Mục tiêu

Tài liệu này quy định **cách chia dữ liệu và cách tính số liệu** cho cả nhóm, để mọi người dùng cùng một cách đo và kết quả so sánh được với nhau.

Gồm 4 phần việc:

- **Segmentation:** đo chất lượng mask bằng IoU, Dice, pixel precision, pixel recall.
- **Detection:** đo bằng precision, recall, F1 và AP.
- **Tốc độ:** đo FPS và latency, ghi rõ phần cứng và cấu hình.
- **Crop và điều kiện ảnh:** xem lỗi detection và điều kiện ảnh (tối, mờ...) ảnh hưởng thế nào đến mask. Phần so sánh GT crop với predicted crop làm ở bước sau (mục 3.5).

Hai câu hỏi chính được nhắc lại ở các mục sau:

- **Q1:** RetinaFace tốt hơn Haar Cascade bao nhiêu? (detection, WIDER FACE)
- **Q2:** Trong 3 loss BCE, Dice, BCE+Dice, loss nào tốt hơn? (segmentation, CelebAMask-HQ, đo trên validation)

## 2. Dữ liệu và cách chia tập

### 2.1. Segmentation – CelebAMask-HQ

- **Cách chia:** ngẫu nhiên theo ảnh, **70% train / 15% validation / 15% test**, seed cố định **42**. Ví dụ 1.000 ảnh thì 700 / 150 / 150.
- **Manifest:** lưu danh sách ảnh của từng tập vào một file. Các lần chạy sau dùng lại đúng file này, không chia lại.
- **Vai trò từng tập:**
  - Train: để học.
  - Validation: để chọn checkpoint và so sánh các loss.
  - Test: chỉ chạy **một lần ở cuối**, sau khi đã khóa model, loss, checkpoint và ngưỡng.
- **Giới hạn:** chưa biết CelebAMask-HQ có thông tin "ảnh nào là cùng một người" hay không. Nếu chia theo ảnh thì một người có thể nằm ở cả train và test. Nếu có thông tin này thì chia theo người. Việc kiểm tra không chặn tiến độ.
- **Q2:** so sánh 3 loss trên **validation**. Test chỉ chạy cho model cuối cùng. Nếu đưa bảng so sánh loss vào báo cáo thì ghi rõ đó là số trên validation. *Hương xác nhận.*

### 2.2. Detection – WIDER FACE

- RetinaFace pretrained đã được train trên tập train của WIDER FACE. Nhãn test chính thức thì không công khai.
- Vì vậy dùng **tập validation chính thức của WIDER FACE làm tập test detection**. Không dùng tập này để chọn checkpoint hay chỉnh evaluator.
- **Tập tune cho Haar:** chọn `scaleFactor`, `minNeighbors`, `minSize` trên **500 ảnh cố định lấy từ WIDER FACE train** (seed 42, lưu danh sách vào manifest riêng). Tập này chỉ để tune tham số Haar và chọn ngưỡng score của Haar, không dùng để báo cáo kết quả. *Uyên xác nhận.*

### 2.3. Hai bộ dữ liệu không thay thế nhau

CelebAMask-HQ không có bbox. WIDER FACE không có mask. Không dùng nhãn của bộ này làm đáp án cho bộ kia.

## 3. Đánh giá segmentation

### 3.1. Công thức

Mask xác suất được chuyển thành mask 0/1 bằng ngưỡng **0.5**. Với mỗi pixel:

- **TP:** đoán là mặt, thật sự là mặt.
- **FP:** đoán là mặt, nhưng thật sự là nền.
- **FN:** đoán là nền, nhưng thật sự là mặt (bị bỏ sót).

| Metric | Công thức |
|---|---|
| IoU | `TP / (TP + FP + FN)` |
| Dice | `2TP / (2TP + FP + FN)` (cũng bằng `2·IoU / (1 + IoU)`) |
| Pixel precision | `TP / (TP + FP)` (có tô thừa không) |
| Pixel recall | `TP / (TP + FN)` (có sót mặt không) |

**Trường hợp mask rỗng** (không có pixel mặt):

| Prediction | Ground truth | IoU, Dice | Precision, recall |
|---|---|---|---|
| rỗng | rỗng | 1 | 1 |
| rỗng | có mặt | 0 | 0 |
| có mặt | rỗng | 0 | 0 |

Quy tắc chung: nếu mẫu số bằng 0 thì kết quả là 1 khi cả hai bên đều rỗng, còn lại là 0. Đây là quy ước đề xuất, *Hương xác nhận*. File `seg_metrics.py` phải làm theo quy ước này, không phải ngược lại.

### 3.1b. Tính metric ở không gian nào

Mọi metric segmentation tính trong **không gian ảnh gốc**, so với mask ground truth gốc.

Nếu model chạy ở kích thước cố định (resize hoặc crop), làm theo thứ tự:

1. Resize **bản đồ xác suất** về kích thước tương ứng.
2. Mới nhị phân hóa bằng ngưỡng 0.5.
3. Nếu có crop, dán mask vào ảnh trống cỡ ảnh gốc, tại vị trí `(x1, y1)`.

Không tính metric trong không gian crop hoặc không gian đã resize, vì sẽ không so sánh được giữa các cấu hình.

Khi đánh giá U-Net độc lập trên test CelebAMask-HQ (chưa qua detection), dùng ảnh và mask ở kích thước gốc của dataset. Chỉ so sánh các lần chạy khi **cùng không gian tính và cùng định nghĩa mask**. *Hương xác nhận.*

### 3.2. Macro và micro

- **Macro (số chính):** tính metric cho từng ảnh rồi lấy thống kê. Báo cáo mean, std và median của IoU và Dice, kèm pixel precision và recall.
- **Micro (số bổ sung):** cộng TP, FP, FN của toàn bộ pixel rồi mới tính một lần. Không lấy trung bình metric từng ảnh để thay cho micro.

### 3.3. Ví dụ tính tay

Mask `4 × 4`: ground truth có 4 pixel mặt, prediction có 4 pixel mặt, trùng nhau 2 pixel.

- `TP = 2`, `FP = 2`, `FN = 2`
- `IoU = 2 / (2 + 2 + 2) = 1/3`
- `Dice = 4 / (4 + 2 + 2) = 0.5`
- `Precision = 2/4 = 0.5`, `Recall = 2/4 = 0.5`

### 3.4. Mask "mặt" gồm những lớp nào

**Chưa chốt, Huy phụ trách.** Dùng tên lớp, không dùng ID số, và đối chiếu README của dataset.

Đề xuất chờ Huy duyệt:

- **Gộp:** `skin, nose, l_eye, r_eye, l_brow, r_brow, mouth, u_lip, l_lip`
- **Không gộp:** `hair, hat, eye_g, l_ear, r_ear, ear_r, neck, neck_l, cloth, background`
- **Cần quyết định thêm:** có gộp tai và kính không.

Chỉ so sánh các metric khi các lần chạy dùng cùng một định nghĩa mask.

### 3.5. So sánh crop và điều kiện ảnh

Làm ở bước sau. Sẽ so sánh **GT crop** với **predicted crop** để xem lỗi detection làm mask tệ đi bao nhiêu, và khảo sát điều kiện ảnh (tối, mờ, mặt nhỏ...). Cách làm chi tiết và các nhóm điều kiện sẽ thống nhất trước khi tổng hợp kết quả.

## 4. Đánh giá detection

### 4.1. Cách ghép prediction với ground truth

Dùng `detect_metrics.py` của Uyên. **Không dùng evaluator chính thức của WIDER FACE.**

- Bbox có dạng `[x1, y1, x2, y2]` trên ảnh gốc. `x2`, `y2` là tọa độ exclusive.
- Ghép prediction với ground truth theo thứ tự score giảm dần.
- Mỗi ground truth chỉ được ghép tối đa một lần.
- Ngưỡng IoU: **0.5**.
- Prediction ghép được là TP. Prediction không ghép được là FP. Ground truth hợp lệ chưa được ghép là FN.
- Mặt có `invalid=1` bị loại khỏi ground truth. Prediction khớp với mặt invalid thì bỏ qua, không tính là FP.
- **Không loại mặt nhỏ.** Báo thêm kết quả theo 3 nhóm chiều cao bbox: nhỏ, vừa, lớn. *Uyên chốt ngưỡng chiều cao.*

### 4.2. Công thức

- `Precision = TP / (TP + FP)`
- `Recall = TP / (TP + FN)`
- `F1 = 2PR / (P + R)`; nếu `P + R = 0` thì `F1 = 0`

Nếu không có prediction hoặc không có ground truth hợp lệ: cả hai rỗng thì P, R, F1 bằng 1; ngược lại bằng 0 (giống mục 3.1).

### 4.3. Chọn ngưỡng score và so sánh RetinaFace với Haar (Q1)

Đề xuất mặc định:

| Detector | Ngưỡng score | Lý do |
|---|---|---|
| RetinaFace | **0.5** cố định | Model đã thấy tập WIDER train khi học, nên không tune ngưỡng trên tập train. |
| Haar | Chọn điểm **F1 cao nhất** trên tập tune 500 ảnh (mục 2.2) | Haar không train trên WIDER nên tune được. |

Khóa cả hai ngưỡng **trước khi** chạy WIDER val. Phép so sánh chính cho Q1 là Precision, Recall, F1 tại ngưỡng đã khóa, trên cùng tập test, cùng IoU 0.5, cùng quy tắc `invalid`.

**Haar không có score sẵn.** Đề xuất lấy `levelWeights` từ `detectMultiScale3(..., outputRejectLevels=True)` làm score, nếu xác minh được.

- Nếu **không** lấy được score: không báo AP cho Haar, và dùng bộ tham số `scaleFactor`, `minNeighbors`, `minSize` cho F1 cao nhất trên tập tune làm điểm hoạt động.
- Việc cần xác minh (*Uyên*): kiểm tra `cv2.__version__`, `help(cascade.detectMultiScale3)` và chạy thử để xác nhận hàm trả về `rects`, `rejectLevels`, `levelWeights` và `len(rects) == len(rejectLevels) == len(levelWeights)`.

### 4.4. Cách tính AP

Khi tính AP, chạy RetinaFace với `conf_thres` thấp (đề xuất **0.01**) để giữ gần như toàn bộ prediction. Giữ `nms_thres = 0.4` cho mọi lần chạy. Nếu để mặc định 0.5, đường precision-recall bị cắt và AP thấp hơn thực tế. Sau khi tính AP xong mới lọc `score >= ngưỡng đã khóa` để tính Precision, Recall, F1. *Uyên xác nhận.*

Dùng **all-point interpolation**:

1. Tại mỗi mức recall, lấy precision lớn nhất ở các mức recall bằng hoặc cao hơn mức đó (gọi là **precision envelope**).
2. Tính diện tích dưới đường precision-recall.

Kết quả chỉ có giá trị trong giao thức của nhóm, **không so sánh trực tiếp với bảng xếp hạng WIDER FACE**. Nếu Haar dùng `levelWeights`, AP của Haar chỉ để tham khảo, vì score của hai detector không cùng thang. Không so trực tiếp AP của Haar với AP của RetinaFace.

**Ví dụ:** có 4 mặt thật hợp lệ và 5 prediction theo thứ tự `TP, TP, FP, TP, FP`. Recall tăng theo bước 0.25. Sau khi lấy precision envelope, bốn khoảng recall `[0, 0.25]`, `[0.25, 0.5]`, `[0.5, 0.75]`, `[0.75, 1]` có precision lần lượt là `1, 1, 0.75, 0`. Khoảng cuối bằng 0 vì recall không bao giờ đạt 1.

`AP = 0.25 × (1 + 1 + 0.75 + 0) = 0.6875`

## 5. Đo FPS và latency

- Chạy **warm-up** trước khi đo, và không tính warm-up vào kết quả.
- Nếu chạy GPU, đồng bộ GPU ngay trước khi bắt đầu bấm giờ và ngay sau khi inference xong.
- Đo tối thiểu **100 lượt**.
- Báo cáo **latency trung bình** và **p95**.
- Ghi rõ: phần cứng, batch size, độ phân giải, FP32 hay FP16, số lượt đo, và phạm vi tính giờ (chỉ model, hay gồm cả tiền xử lý và hậu xử lý như crop, map mask, blur).

**Ví dụ:** 100 ảnh chạy hết 5 giây thì `FPS = 100 / 5 = 20 ảnh/giây` và latency trung bình `= 5000 / 100 = 50 ms/ảnh`. Nếu đo được p95 là 80 ms thì báo riêng 80 ms, không suy ra p95 từ FPS.

Người phụ trách đo: **Việt**.