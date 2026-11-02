# Bàn giao UY-1 (Task 1 Detection): RetinaFace inference (TV2-UYEN)

## 1. Tóm tắt 

Task này xây dựng bước phát hiện khuôn mặt để pipeline của nhóm có thể đi từ ảnh đầu vào đến danh sách các vùng mặt. Module dùng RetinaFace pretrained; **không huấn luyện lại model**.

Luồng dự kiến:

```text
Ảnh BGR -> RetinaFace -> bbox + score + 5 landmarks -> crop mặt -> U-Net
```

Phần blur/che mặt và segmentation thuộc các task tích hợp tiếp theo, không nằm trong module detection này.

## 2. Vì sao cần làm 

Detection là bước đầu của pipeline: cần xác định khuôn mặt và tọa độ bbox trước khi crop vùng mặt để đưa vào U-Net segmentation. Nếu chưa có bbox đúng, bước crop và đánh giá độ bền của crop cũng không có đầu vào phù hợp.

Task dùng model RetinaFace pretrained thay vì train lại vì phạm vi task 1 detection là xây dựng inference adapter; nhóm có thể dùng detector đã huấn luyện sẵn và tập trung vào tích hợp, kiểm thử hợp đồng dữ liệu, và đánh giá trên dữ liệu dự án. Code của task không có bước huấn luyện.

FaceXLib được chọn vì dùng PyTorch, nhận ảnh OpenCV và hỗ trợ truyền ngưỡng NMS. Bảng so sánh các lựa chọn pretrained được giữ ở phần triển khai.

## 3. File và vị trí trong repo 

Các file thuộc phạm vi task 1 detection:

- `src/detection/retinaface.py`: detector, chuẩn hóa input/output, `bbox_to_int`, batch, vẽ và CLI.
- `tests/test_retinaface.py`: unit test hợp đồng và các trường hợp ảnh lỗi.
- `requirements.txt`: dependency của project.
- `docs/detection/TV2-UYEN_retinaface1.md`: tài liệu bàn giao task.
- `.gitignore`: loại weights, dataset và output sinh ra khỏi Git.

### Đã làm

- Tạo `RetinaFaceDetector` trong [`src/detection/retinaface.py`](../../src/detection/retinaface.py).
- Chọn **FaceXLib với backbone `resnet50`**. FaceXLib dùng PyTorch, nhận ảnh OpenCV và trả bbox, score cùng landmarks. So với `retina-face` của serengil, lựa chọn này cho phép truyền ngưỡng NMS vào API inference.
- Chuẩn hóa ảnh BGR, ảnh xám và ảnh BGRA trước khi inference. Nếu ảnh quá lớn, module resize để giới hạn kích thước; nếu ảnh rất nhỏ thì upscale. Bbox và landmarks được đổi lại tọa độ ảnh gốc và clip trong biên ảnh.
- Chuẩn hóa kết quả thành danh sách dict chung cho các module trong nhóm. Không phát hiện thấy mặt thì trả `[]`.
- Bổ sung `detect_batch` (chạy từng ảnh theo thứ tự), `draw` để vẽ bbox/landmarks và CLI chạy demo.
- Thêm test cho hợp đồng dữ liệu trong [`tests/test_retinaface.py`](../../tests/test_retinaface.py). Test dùng backend giả lập để chạy nhanh, không phải tải model mỗi lần chạy.
- Khai báo dependency trong [`requirements.txt`](../../requirements.txt).


## 4. Lý thuyết cần biết

RetinaFace dự đoán bbox và 5 landmarks trong cùng một model (multi-task). Bbox xác định vùng mặt; landmarks giúp các bước sau định vị mắt, mũi và miệng để crop/alignment.

Model dùng anchor làm các khung tham chiếu tại nhiều kích thước/vị trí. FPN (Feature Pyramid Network) kết hợp đặc trưng ở nhiều độ phân giải để hỗ trợ nhận diện mặt lớn và nhỏ. Sau khi tạo nhiều bbox ứng viên, NMS (Non-Maximum Suppression) loại các bbox trùng nhau. IoU (Intersection over Union) đo phần giao nhau giữa hai bbox: ví dụ IoU bằng `0.8` lớn hơn ngưỡng NMS `0.4` thì thường chỉ giữ bbox có score cao hơn.

Tăng `conf_thres` thường làm kết quả ít hơn và chắc chắn hơn, nhưng có thể bỏ sót mặt. Giảm ngưỡng có thể tìm được thêm mặt nhỏ/yếu nhưng cũng tăng false positive. `nms_thres` điều chỉnh mức chồng lấp để xem hai bbox có bị coi là trùng hay không.

## 5. Cách làm và lệnh chạy

### Hợp đồng đầu vào/đầu ra

`detect(image_bgr)` nhận ảnh dạng `numpy.ndarray`. Ảnh OpenCV BGR là đầu vào khuyến nghị; ảnh xám và BGRA cũng được hỗ trợ.

Mỗi khuôn mặt được trả về theo dạng:

```python
{
    "bbox": [10.0, 15.0, 50.0, 60.0],
    "score": 0.95,
    "landmarks": [
        [22.0, 30.0],
        [38.0, 30.0],
        [30.0, 40.0],
        [24.0, 50.0],
        [36.0, 50.0],
    ],
}
```

Tọa độ tính bằng pixel trên **ảnh gốc**, gốc tọa độ ở trên-trái. Bbox được clip trong `[0, W] × [0, H]`; `x2` và `y2` là biên exclusive phù hợp với slicing NumPy. Ví dụ bbox `[10, 15, 50, 60]` crop bằng `image[15:60, 10:50]`. Nếu làm tròn/clamp khiến vùng crop có chiều rộng hoặc chiều cao bằng 0, `bbox_to_int` trả `None` và `detect` loại face đó.

Landmarks được clip theo tọa độ pixel hợp lệ `[0, W-1] × [0, H-1]`, khác với bbox exclusive. Các face trong danh sách kết quả được sắp theo `score` giảm dần.

Hàm dùng chung `bbox_to_int(bbox, image_shape)` làm tròn `x1/y1` xuống, `x2/y2` lên, clamp vào `[0, W] × [0, H]` và trả `None` nếu crop rỗng:

```python
from src.detection.retinaface import bbox_to_int

crop_box = bbox_to_int(face["bbox"], image.shape)
if crop_box is not None:
    x1, y1, x2, y2 = crop_box
    face_crop = image[y1:y2, x1:x2]
```

#### Quy tắc chuẩn hóa ảnh

- Ảnh float trong `[0, 1]` được nhân `255`; float ngoài miền này và các kiểu số nguyên khác được clip về `[0, 255]`.
- Ảnh `uint16` được chia `257` để đổi dải `[0, 65535]` về `[0, 255]`.
- Ảnh xám chuyển thành 3 kênh; ảnh BGRA bỏ alpha và giữ thứ tự màu BGR.
- Ảnh một pixel là hợp lệ, được upscale để cạnh ngắn tối thiểu là 32 pixel.
- Ảnh hợp lệ nhưng không phát hiện mặt trả `[]`.

#### Đầu vào ảnh không hợp lệ

`detect()` raise `ValueError` với thông báo rõ trong các trường hợp sau:

| Đầu vào | Hành vi |
|---|---|
| `None` hoặc object không phải `numpy.ndarray` (ví dụ Python list) | `ValueError` |
| Mảng rỗng | `ValueError` |
| Shape 1D, 5D hoặc số kênh 2 | `ValueError` |
| Dtype bool, complex hoặc object | `ValueError` |
| Ảnh có NaN hoặc infinity | `ValueError` |
| Float, số nguyên, gồm `uint16`, với shape ảnh được hỗ trợ | Được chuẩn hóa rồi inference |
| Ảnh hợp lệ không có mặt | Trả `[]` |

### Cài đặt dependency và chạy demo

Cài các dependency trong cùng Python interpreter sẽ dùng để chạy:

```text
python -m pip install -r requirements.txt
```

Python khuyến nghị cho thành viên: **Python 3.12**. Mã nguồn yêu cầu cú pháp Python 3.10 trở lên. Đã tạo môi trường sạch Python 3.12.10, cài requirements, chạy pytest và demo CPU trên ảnh trắng tổng hợp tạm thời. Môi trường inference cũng đã chạy được trên Python 3.14.5 với PyTorch CPU.

Chạy demo và lưu ảnh có vẽ bbox/landmarks:

```text
python -m src.detection.retinaface --image data/raw/sample.jpg --out outputs/retinaface_debug.jpg
```

Có thể chọn thiết bị hoặc đổi confidence threshold:

```text
python -m src.detection.retinaface --image data/raw/sample.jpg --out outputs/debug.jpg --device cpu --conf-thres 0.6
```

Weights pretrained được tải tự động khi khởi tạo model lần đầu; vì vậy lần chạy đầu cần có kết nối mạng và quyền ghi vào thư mục weights của FaceXLib. Với GPU NVIDIA, cần cài PyTorch tương ứng với CUDA trên máy; `device="auto"` sẽ dùng CUDA nếu PyTorch nhận ra GPU, nếu không thì dùng CPU.

Chạy test:

```text
python -m pytest tests/test_retinaface.py -q
```

### Cách TV5 tích hợp

```python
import cv2
from src.detection.retinaface import RetinaFaceDetector, bbox_to_int

image = cv2.imread("data/raw/sample.jpg")
detector = RetinaFaceDetector(device="auto")
faces = detector.detect(image)
debug_image = detector.draw(image, faces)
cv2.imwrite("outputs/retinaface_debug.jpg", debug_image)
```

Ví dụ crop các mặt sau khi detection:

```python
for face in faces:
    crop_box = bbox_to_int(face["bbox"], image.shape)
    if crop_box is not None:
        x1, y1, x2, y2 = crop_box
        face_crop = image[y1:y2, x1:x2]
```

### So sánh các lựa chọn pretrained

Đã cân nhắc nhanh các lựa chọn pretrained sau:

| Thư viện | Điểm phù hợp | Điểm cần lưu ý |
|---|---|---|
| **FaceXLib** | PyTorch; có backbone ResNet50 và MobileNet0.25; nhận ảnh BGR; API trả bbox, score, 5 landmarks và cho truyền ngưỡng NMS | Tự tải weights ở lần đầu; weights cần được lưu bên ngoài Git |
| Pytorch_Retinaface của biubug6 | PyTorch; có mã inference và pretrained weights | Thường cần tự ghép phần decode/NMS và chuẩn hóa đầu ra; cần kiểm tra license của repo/weights trước khi phân phối |
| InsightFace | Có pipeline detection và hỗ trợ CPU/GPU tùy backend | License của code và license model/weights không giống nhau; cần kiểm tra model pack cụ thể |
| `retina-face` của serengil | Cài qua pip; mặc định dùng ONNX Runtime, có thể chọn backend khác | API trả dict theo định dạng riêng; API inference đã kiểm tra dùng NMS mặc định cố định |

FaceXLib được chọn vì hợp với stack PyTorch của nhóm và có API phù hợp hơn với hợp đồng dùng chung. README FaceXLib ghi license MIT cho thư viện và liệt kê nguồn RetinaFace của họ là MIT; nếu dự án có mục đích phân phối ngoài phạm vi môn học, nhóm vẫn nên kiểm tra license của đúng phiên bản weights được sử dụng.

Nguồn tham khảo: [FaceXLib](https://github.com/xinntao/facexlib), [Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface), [InsightFace](https://github.com/deepinsight/insightface), [`retina-face`](https://github.com/serengil/retinaface).

## 6. Kết quả và số liệu

- `python -m pytest tests/test_retinaface.py -q`: **28 tests passed** trên cả Python `3.14.5` và môi trường sạch Python `3.12.10`.
- Demo CLI chạy từ môi trường sạch Python 3.12 trên ảnh trắng tổng hợp tạo tạm trong thư mục Temp; model khởi tạo thành công, phát hiện `0` mặt và ghi file output thành công. Ảnh tổng hợp này không phải ảnh minh chứng accuracy.
- Đã khởi tạo pretrained model và chạy inference trên CPU; ảnh trắng `512 x 512` và ảnh rất nhỏ không có detection.
- Test ảnh 1 mặt, nhiều mặt, clipping, landmarks và scale dùng backend giả lập. Những test đó xác nhận adapter và hợp đồng dữ liệu, **không đo accuracy của model thật**.
- **Ảnh thật (3 ảnh local, không đưa vào repo):** chạy RetinaFace thật (FaceXLib ResNet50) trên CPU Intel Core i7-1255U, Windows 11, batch 1.

| Ảnh | Kích thước ảnh | conf-thres | Số mặt phát hiện | Số mặt đếm bằng mắt | Thời gian TB/ảnh |
|---|---|---|---|---|---|
| Nhiều mặt | 516×387 | 0.5 | 6 | 6 | 0.312 s |
| Đông người | 820×405 | 0.5 | 44 | Không có ground truth | 0.818 s |
| Đông người | 820×405 | 0.3 | 47 | Không có ground truth | 0.515 s |
| Một mặt | 1151×768 | 0.5 | 1 | 1 | 1.767 s |

- Thời gian đo với 2 lần warm-up và 10 lượt đo, batch 1. Thời gian phụ thuộc vào kích thước ảnh nên không so sánh trực tiếp giữa các ảnh.
- Ảnh đông người không có nhãn nên chưa xác định được số mặt nhỏ bị sót hoặc số khung sai. Hạ conf-thres từ 0.5 xuống 0.3 thêm 3 khung; khi xem ảnh debug, các khung thêm trông nằm trên vùng mặt và không thấy rõ khung nào nằm trên vùng chắc chắn không phải mặt. Đây chỉ là nhận xét bằng mắt, không thay thế ground truth. Chênh lệch thời gian giữa hai ngưỡng nằm trong dao động đo, không kết luận ngưỡng làm inference nhanh hơn.
- **Kiểm tra nhánh resize:** ảnh một mặt chạy với max_size=600 để đi qua nhánh resize; bbox lệch tối đa 4.5 pixel theo từng tọa độ so với inference không resize. Đây là độ lệch giữa hai lần inference, không phải sai số so với ground truth; chỉ là kiểm tra tính nhất quán sơ bộ. Chưa kiểm tra ảnh gốc có cạnh dài trên 2150 pixel.
- Các số liệu trên chỉ là thử nghiệm trên ảnh thật cỡ nhỏ, không phải AP trên tập chuẩn (thuộc UY-3).
- Số đo đã có từ lần chạy trước chỉ là smoke benchmark trên ảnh trắng tổng hợp `512 x 512`, batch 1, CPU Intel Core i7-1255U, Windows 11: lượt đầu khoảng `0.476 s`; sau 2 warm-up, trung bình 5 lượt khoảng `0.413 s/ảnh`. Đây không phải benchmark ảnh có mặt và không thay thế yêu cầu đo trên ảnh thật.
- Môi trường sạch: Python `3.12.10`, FaceXLib `0.3.0`, PyTorch `2.15.0+cpu`, torchvision `0.29.2+cpu`, NumPy `2.5.4`, OpenCV `5.0.0`, pytest `9.2.0`; Windows 11, CPU Intel Core i7-1255U, CUDA không khả dụng.
- Môi trường sẵn có: Python `3.14.5`, FaceXLib `0.3.0`, PyTorch `2.15.0+cpu`, torchvision `0.29.2+cpu`, NumPy `2.5.4`, OpenCV `5.0.0`, pytest `9.2.0`.
- Đã kiểm tra `.gitignore`: `data/raw/`, `data/processed/`, `data/splits/`, `outputs/`, `models/`, `weights/`, `facexlib/weights/` và virtualenv bị ignore. `git status` không có ảnh dataset hay file weights trong danh sách thay đổi.
- Không dùng seed vì không train hoặc sinh dữ liệu ngẫu nhiên trong inference/test.

## 8. Checklist Definition of Done
97
- [x] Code chạy từ đầu trên môi trường sạch Python 3.12.10 (pytest và demo CPU trên ảnh tổng hợp).
- [x] Pytest pass.
- [x] Ảnh thật đã được kiểm tra (3 ảnh local, xem mục 6).
- [x] Doc đủ 8 mục.
- [x] Nhánh riêng `feature/tv2-uyen-detection`, không đụng `main`.
- [ ] Đã tag người review.
- [ ] Đã cập nhật Backlog.

##  Link PR/commit
8 . 
- Link PR: 
- Commit: 
