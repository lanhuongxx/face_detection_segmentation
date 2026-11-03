# Bàn giao UY-2 (Baseline Detection): Haar Cascade (TV2-UYEN)

## 1. Tóm tắt

Task này bổ sung baseline phát hiện mặt Haar Cascade để so sánh với RetinaFace trên cùng manifest. Detector dùng cascade `haarcascade_frontalface_default.xml` có sẵn trong OpenCV, chạy CPU và không huấn luyện model.

Luồng:

```text
CSV manifest -> đọc ảnh BGR -> Haar Cascade -> bbox + score -> predictions JSON
```

`HaarFaceDetector.detect(image_bgr)` trả danh sách khuôn mặt theo hợp đồng chung: bbox pixel trong ảnh gốc, score trong `[0, 1]`, `landmarks=None`. Ảnh hợp lệ không phát hiện mặt trả `[]`.

Người phụ trách task: Uyên. Nguyên (TV5) phụ trách tích hợp; Hương quản lý repo. Schema manifest hiện dùng tạm `image_path,split`, chờ Huy xác nhận. Đã chạy thử trên 3 ảnh local (mục 6); test trên ảnh mặt frontal chuẩn vẫn skip vì chưa có fixture; còn chờ review.

## 2. Vì sao cần làm

Haar là baseline cổ điển, nhẹ và chạy CPU, cung cấp mốc so sánh đơn giản với RetinaFace. Chạy hai detector trên đúng cùng manifest và split giúp phép so sánh công bằng hơn; đánh giá AP chính thức vẫn cần ground truth và thuộc bước evaluation.

Haar Cascade mặc định tập trung vào khuôn mặt nhìn gần chính diện. Đây là giới hạn hữu ích để đối chiếu với detector hiện đại, đồng thời cho nhóm một lựa chọn inference nhẹ khi kiểm thử pipeline.

## 3. File và vị trí trong repo

- [`src/detection/haar_baseline.py`](../../src/detection/haar_baseline.py): `HaarFaceDetector`, kiểm tra ảnh, tiền xử lý grayscale/equalizeHist và chuẩn hóa kết quả.
- [`src/detection/run_detect.py`](../../src/detection/run_detect.py): đọc manifest, gọi Haar hoặc RetinaFace, ghi predictions JSON.
- [`configs/detection.yaml`](../../configs/detection.yaml): cấu hình hai detector.
- [`tests/test_haar_baseline.py`](../../tests/test_haar_baseline.py): kiểm thử hợp đồng, score, tiền xử lý, manifest và output JSON.
- [`requirements.txt`](../../requirements.txt): giới hạn OpenCV ở phiên bản 4.x và thêm PyYAML.

### Đã làm

- Dùng `cv2.CascadeClassifier` với `cv2.data.haarcascades/haarcascade_frontalface_default.xml`.
- Dùng `detectMultiScale3(..., outputRejectLevels=True)` để nhận `levelWeights`; đổi weight `w` thành score bằng sigmoid `1 / (1 + exp(-w))`.
- Hỗ trợ cấu hình `scale_factor`, `min_neighbors`, `min_size` và `equalize_hist` trong YAML.
- Chuẩn hóa ảnh BGR, BGRA hoặc grayscale về ảnh xám trước khi dò; không phát hiện mặt trả `[]`.
- Bổ sung CLI để Haar và RetinaFace dùng chung CSV, cùng chọn split và ghi schema JSON `{image_id: [{"bbox": [...], "score": ...}]}`.
- Thêm test integration của luồng Haar trên ảnh trắng. Test phát hiện mặt frontal dùng ảnh thật qua biến môi trường `UY2_FACE_IMAGE`; hiện chưa có fixture nên test đó skip.

## 4. Lý thuyết cần biết

Viola–Jones dùng Haar-like features để đo chênh lệch tổng cường độ giữa các vùng sáng/tối. Integral image lưu tổng tích lũy để tính tổng trên một vùng bằng một số ít phép toán. AdaBoost chọn và kết hợp các feature hữu ích; cascade gồm nhiều tầng, loại nhanh vùng nền trước khi thực hiện các tầng tiếp theo.

Mặt nghiêng làm thay đổi hình dạng biểu kiến so với mẫu frontal; ảnh tối hoặc tương phản thấp làm yếu tín hiệu sáng/tối; mặt nhỏ không đủ pixel để feature phân biệt ổn định. Vì vậy Haar dễ bỏ sót các trường hợp này hoặc tạo false positive.

`levelWeight` biểu thị mức độ phù hợp của detection với cascade, không phải xác suất đã hiệu chuẩn. Sigmoid là hàm đơn điệu nên giữ thứ tự xếp hạng: ví dụ `w=-2, 0, 2` tương ứng score khoảng `0.119, 0.500, 0.881`. Score này **chỉ dùng để xếp hạng khi tính AP, không phải xác suất thật**.

## 5. Cách làm và lệnh chạy

### Hợp đồng đầu vào/đầu ra

`detect(image_bgr)` nhận ảnh `numpy.ndarray` uint8 dạng BGR; ảnh grayscale, một kênh và BGRA cũng được hỗ trợ. Ảnh không hợp lệ sẽ báo `ValueError`; cascade XML không tải được sẽ báo `FileNotFoundError`.

Mỗi detection có dạng:

```python
{
    "bbox": [10.0, 15.0, 50.0, 60.0],
    "score": 0.88,
    "landmarks": None,
}
```

`bbox` dùng pixel trong ảnh gốc, gốc tọa độ góc trên-trái; `x2/y2` là biên exclusive giống slicing NumPy. Ví dụ `[10, 15, 50, 60]` tương ứng `image[15:60, 10:50]`. Kết quả được sắp theo score giảm dần.

CLI bỏ trường `landmarks` để giữ schema predictions chung:

```json
{
  "images/person_001.jpg": [
    {"bbox": [10.0, 15.0, 50.0, 60.0], "score": 0.88}
  ],
  "images/person_002.jpg": []
}
```

### Manifest và cấu hình

Schema manifest tạm thời:

```csv
image_path,split
images/person_001.jpg,test
images/person_002.jpg,test
```

Đường dẫn `image_path` tương đối được tính từ thư mục chứa CSV. Khi Huy cung cấp schema chính thức, cần xác nhận và cập nhật loader/test nếu khác. Manifest đang dùng `image_path` làm `image_id`; vì vậy đường dẫn này phải duy nhất trong split được chạy.

Cấu hình Haar trong [`configs/detection.yaml`](../../configs/detection.yaml):

```yaml
haar:
  scale_factor: 1.1
  min_neighbors: 5
  min_size: [30, 30]
  equalize_hist: false
```

| Tham số | Mặc định | Ảnh hưởng cần quan sát |
|---|---:|---|
| `scale_factor` | `1.1` | Giá trị gần 1 quét nhiều scale dày hơn, thường chậm hơn và có thể bắt được mặt ở nhiều kích cỡ; tăng giá trị thường nhanh hơn nhưng có thể bỏ mặt nhỏ. |
| `min_neighbors` | `5` | Tăng giá trị cần nhiều cửa sổ lân cận đồng thuận, thường giảm false positive nhưng có thể bỏ detection yếu; giảm giá trị thường tăng detection và false positive. |
| `min_size` | `[30, 30]` | Kích thước cửa sổ nhỏ nhất theo `[width, height]`; tăng ngưỡng bỏ mặt nhỏ và giảm số vùng cần dò. |
| `equalize_hist` | `false` | Bật cân bằng histogram ảnh xám có thể làm rõ ảnh tương phản thấp, nhưng cũng khuếch đại nhiễu hoặc làm giảm kết quả trên ảnh vốn đã rõ. |

Benchmark ảnh local ở mục 6 dùng `min_size=[30, 30]`, `equalize_hist=false` và lần lượt thử mọi tổ hợp `scale_factor` / `min_neighbors` trong bảng. Mỗi ảnh/cấu hình chạy 2 lượt warm-up, sau đó đo 10 lượt; ảnh không được sao chép vào repo.

### Cài đặt dependency và chạy

Cài dependencies bằng đúng Python interpreter dùng để chạy:

```powershell
python -m pip install -r requirements.txt
```

`opencv-python<5` giới hạn dependency trong major 4; repo hiện chưa ghi lý do tương thích cụ thể, nên cần hỏi người thêm pin và chạy test với OpenCV 5 để xác nhận trước khi nới giới hạn.

Chạy Haar và RetinaFace trên cùng manifest/split:

```powershell
python -m src.detection.run_detect --detector haar --manifest data\splits\manifest.csv --out outputs\haar.json --split test
python -m src.detection.run_detect --detector retinaface --manifest data\splits\manifest.csv --out outputs\retinaface.json --split test
```

Mặc định dùng `configs/detection.yaml`; có thể đổi bằng `--config`. Không truyền `--split` thì xử lý mọi dòng manifest. Lần đầu khởi tạo RetinaFace có thể cần tải pretrained weights; Haar không cần weights ngoài XML đi kèm OpenCV.

Chạy test:

```powershell
python -m pytest -q
```

Test với ảnh mặt frontal thật:

```powershell
$env:UY2_FACE_IMAGE = "data\raw\frontal_face.jpg"
python -m pytest tests\test_haar_baseline.py -q
```

### Cách TV5 tích hợp

```python
import cv2
from src.detection.haar_baseline import HaarFaceDetector
from src.detection.retinaface import bbox_to_int

image = cv2.imread("data/raw/sample.jpg")
detector = HaarFaceDetector()
faces = detector.detect(image)

for face in faces:
    crop_box = bbox_to_int(face["bbox"], image.shape)
    if crop_box is not None:
        x1, y1, x2, y2 = crop_box
        face_crop = image[y1:y2, x1:x2]
```

Nếu cần chạy benchmark/export predictions, dùng `run_detect.py` để Haar và RetinaFace đọc cùng manifest thay vì tự tạo hai danh sách ảnh riêng.

## 6. Kết quả và số liệu

- Kiểm thử toàn bộ repo: `47 passed, 1 skipped`. Test bị skip cần ảnh mặt frontal thật vì hiện chưa có fixture.
- Đã kiểm tra CLI `--help`; module hoạt động trên Windows terminal mặc định sau khi đặt stdout/stderr về UTF-8.
- Môi trường: Python `3.14.5`, OpenCV runtime `4.14.0` (`opencv-python 4.14.0.94`), PyYAML `6.0.3`, NumPy `2.5.4`, pytest `9.2.0`.
- Phần cứng: Windows 11, AMD64, CPU Intel, 12 logical CPUs; CUDA không khả dụng. Haar chạy CPU.
- Không dùng seed vì inference không có bước ngẫu nhiên.
- Haar được chạy trên ba ảnh local gọi là **nhiều mặt**, **đông người**, **một mặt**. Mỗi tổ hợp tham số/ảnh có 2 lượt warm-up và 10 lượt đo; thời gian là trung bình ms/ảnh. Cố định `min_size=[30, 30]`, `equalize_hist=false`.

| `scale_factor` | `min_neighbors` | Nhiều mặt (số / ms) | Đông người (số / ms) | Một mặt (số / ms) |
|---:|---:|---:|---:|---:|
| 1.05 | 3 | 7 / 31.70 | 1 / 57.81 | 10 / 163.44 |
| 1.05 | 5 | 2 / 30.40 | 0 / 55.64 | 6 / 148.15 |
| 1.05 | 8 | 0 / 28.63 | 0 / 57.20 | 5 / 148.38 |
| 1.1 | 3 | 2 / 14.40 | 0 / 27.73 | 8 / 74.85 |
| 1.1 | 5 | 0 / 14.97 | 0 / 34.93 | 5 / 76.11 |
| 1.1 | 8 | 0 / 14.63 | 0 / 29.18 | 4 / 74.10 |
| 1.3 | 3 | 0 / 6.49 | 0 / 13.12 | 4 / 30.85 |
| 1.3 | 5 | 0 / 6.26 | 0 / 14.40 | 4 / 30.92 |
| 1.3 | 8 | 0 / 6.32 | 0 / 12.56 | 1 / 30.54 |

So nhanh, số mặt RetinaFace với cấu hình mặc định trong `configs/detection.yaml` (`device=auto`, chạy CPU; `conf_thres=0.5`, `nms_thres=0.4`): **nhiều mặt: 6**, **đông người: 44**, **một mặt: 1**. Đây **chưa phải đánh giá chính thức, chờ UY-3**. Không có ground truth nên các số đếm này không kết luận detection đúng hay sai, cũng không thay thế đánh giá AP/recall.

- Test unit dùng cascade giả để kiểm tra chuyển đổi output; test luồng thật dùng ảnh trắng tổng hợp và chỉ xác nhận ảnh không có detection trả JSON list rỗng. Cần manifest và ground truth thống nhất để đánh giá AP/recall chính thức.

## 7. Checklist Definition of Done

- [x] `HaarFaceDetector.detect(image_bgr)` cùng schema RetinaFace; `landmarks=None`, ảnh không có mặt trả `[]`.
- [x] Có cấu hình `scale_factor`, `min_neighbors`, `min_size`, `equalize_hist` trong YAML.
- [x] Cùng CLI/manifest tạo predictions theo schema chung cho Haar và RetinaFace.
- [x] Pytest cho output, score, tiền xử lý, ảnh trắng và luồng manifest/JSON.
- [x] Chạy full test suite: `47 passed, 1 skipped`.
- [ ] Chạy test trên ảnh mặt frontal được nhóm duyệt.
- [ ] Huy xác nhận schema manifest chính thức.
- [ ] Một thành viên khác review/approve PR; cập nhật trạng thái task và backlog.
