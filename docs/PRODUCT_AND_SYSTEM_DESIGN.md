# Thiết kế sản phẩm và hệ thống suy luận

> **Tên sản phẩm tạm:** FaceShield
>
> **Dự án:** Phát hiện và phân vùng khuôn mặt trong ảnh đông người
>
> **Công nghệ:** RetinaFace + U-Net + Streamlit
> **Mục đích:** Mô tả trạng thái hiện tại của sản phẩm và hợp đồng dữ liệu giữa các module.

---

## 1. Tóm tắt sản phẩm

FaceShield là ứng dụng suy luận cho phép người dùng tải ảnh hoặc khung hình từ video để:

1. Phát hiện các khuôn mặt bằng RetinaFace.
2. Phân vùng từng khuôn mặt bằng U-Net.
3. Xem ảnh gốc, box phát hiện và mask phân vùng.
4. Tải ảnh phân vùng dạng PNG và báo cáo kết quả dạng JSON.

Mã nguồn hiện tại **không nhận diện danh tính**. Box và mask chỉ mô tả vị trí khuôn mặt, không gắn tên người.

### Trạng thái chức năng

| Chức năng | Trạng thái hiện tại |
|---|---|
| Tải ảnh JPG, JPEG, PNG, WebP | Đã hỗ trợ |
| Chọn ảnh mẫu | Đã hỗ trợ nếu thư mục `samples/` có ảnh |
| Tải video MP4, MOV, AVI, MKV | Đã hỗ trợ để trích khung hình |
| Chọn và phân tích một khung hình video | Đã hỗ trợ |
| Phát hiện nhiều khuôn mặt | Đã hỗ trợ |
| Phân vùng bằng U-Net | Đã hỗ trợ |
| Ellipse dự phòng | Đã hỗ trợ |
| Tải PNG và báo cáo JSON | Đã hỗ trợ |
| Làm mờ / pixel hóa / tô màu từng mặt | Pipeline đã có, giao diện hiện tắt |
| So sánh che theo box và mask | Pipeline đã có, giao diện hiện tắt |
| Ghi/ xuất video đã xử lý | Chưa hỗ trợ |
| Webcam trực tiếp | Chưa hỗ trợ |
| Theo dõi khuôn mặt giữa các khung hình | Chưa hỗ trợ |

Trạng thái này được điều khiển bởi `ENABLE_PRIVACY = False` trong `app.py`. Khi bật riêng tính năng che mặt, các thành phần UI và API privacy filter sẵn có có thể được kích hoạt lại mà không cần chạy lại detection/segmentation.

---

## 2. Người dùng và use case

### 2.1 Nhóm người dùng

| Nhóm người dùng | Nhu cầu |
|---|---|
| Người làm truyền thông | Kiểm tra khuôn mặt trong ảnh sự kiện trước khi chia sẻ |
| Giáo viên / nhà trường | Xem vùng khuôn mặt học sinh trong ảnh hoạt động |
| Nhà nghiên cứu | Kiểm tra đầu ra RetinaFace và U-Net |
| Người phát triển | Chạy pipeline bằng Streamlit hoặc Python |

### 2.2 User stories

| ID | User story | Tiêu chí chấp nhận | Trạng thái |
|---|---|---|---|
| US1 | Tải một ảnh để xem pipeline phát hiện và phân vùng | Hiển thị ảnh gốc, box, mask và số mặt | Đã đạt |
| US2 | Phân tích khung hình từ video | Chọn video, chọn frame, chạy phân tích frame | Đã đạt |
| US3 | Xem các khuôn mặt phát hiện riêng lẻ | Hiển thị thumbnail, ID, score và kích thước | Đã đạt |
| US4 | Tải kết quả phân vùng | Tải được PNG cùng kích thước ảnh | Đã đạt |
| US5 | Tải báo cáo chi tiết | JSON hợp lệ và có số liệu từng mặt | Đã đạt |
| US6 | Xử lý ảnh không có khuôn mặt | Hiển thị thông báo rõ ràng, không crash | Đã đạt |
| US7 | Chọn che từng khuôn mặt hoặc đổi kiểu che | Ảnh kết quả thay đổi chỉ theo lựa chọn | Pipeline đã có; UI tắt |
| US8 | So sánh che box và mask | Hiển thị hai kết quả cạnh nhau | Pipeline đã có; UI tắt |

### 2.3 Phạm vi

**Trong phạm vi:**

- Một ảnh RGB hoặc một khung hình video mỗi lượt.
- Phát hiện và phân vùng nhiều khuôn mặt.
- Giới hạn kích thước ảnh để giảm tải xử lý.
- Fallback ellipse cho mặt quá nhỏ hoặc mask không hợp lệ.
- Ghi và xuất JSON/report.
- Chạy CPU hoặc CUDA.

**Ngoài phạm vi hiện tại:**

- Nhận diện danh tính.
- Theo dõi khuôn mặt qua video.
- Che mặt bắt buộc/không thể khôi phục.
- Xuất video.
- Webcam trực tiếp.
- Lưu trữ hoặc đồng bộ ảnh trên máy chủ.
- Xác thực, tài khoản và quyền truy cập.

---

## 3. Kiến trúc hệ thống

```text
Người dùng
  │
  ├─ app.py (Streamlit)
  │    ├─ upload ảnh / chọn ảnh mẫu
  │    └─ upload video / chọn frame
  │
  ▼
src.pipeline.preprocess_input
  └─ kiểm tra định dạng, dung lượng, RGB, giới hạn max_side
  │
  ▼
src.pipeline.inference.FacePipeline.analyze()
  ├─ src.detection.retinaface.RetinaFaceDetector
  ├─ src.pipeline.crop
  ├─ src.segmentation.unet.FaceSegmenter
  ├─ src.pipeline.mask_mapping
  └─ src.pipeline.visualize
  │
  ▼
PipelineResult
  ├─ ảnh gốc sau tiền xử lý
  ├─ danh sách Face
  ├─ instance_map
  ├─ overlay_boxes
  └─ overlay_masks
  │
  ▼
src.pipeline.inference.FacePipeline.apply_privacy()
  └─ src.pipeline.privacy_filter
  │
  ▼
Ảnh kết quả + báo cáo JSON
```

`app.py` điều phối trạng thái phiên bằng `st.session_state`, nhưng không chứa thuật toán detection/segmentation. Khi đổi kiểu che hoặc vùng che, chương trình gọi lại `apply_privacy()` thay vì chạy lại RetinaFace và U-Net.

---

## 4. Hợp đồng dữ liệu

### 4.1 Quy ước chung

| Hạng mục | Quy ước |
|---|---|
| Ảnh trong bộ nhớ | `numpy.ndarray`, `shape=(H, W, 3)`, `dtype=uint8`, kênh RGB |
| Hệ tọa độ | Pixel trên ảnh sau tiền xử lý, gốc `(0, 0)` ở góc trên trái |
| Bounding box | `(x1, y1, x2, y2)`, số nguyên, đã clip trong ảnh |
| Mask nhị phân | `numpy.ndarray`, `uint8`, giá trị `{0, 1}` |
| Mask xác suất | `float32`, giá trị trong `[0, 1]` |
| ID khuôn mặt | Số nguyên bắt đầu từ 1; các mặt được sắp theo `x1` tăng dần |
| Thiết bị | `cuda` nếu cấu hình `auto` phát hiện CUDA; ngược lại `cpu` |

### 4.2 Dataclass

Các kiểu dữ liệu chính nằm trong `src/pipeline/schemas.py`:

```python
@dataclass
class Detection:
    bbox: tuple
    score: float
    landmarks: np.ndarray | None = None


@dataclass
class Face:
    id: int
    bbox: tuple
    score: float
    landmarks: np.ndarray | None = None
    crop_box: tuple | None = None
    mask_box: tuple | None = None
    mask_local: np.ndarray | None = None
    thumbnail: np.ndarray | None = None
    status: str = "ok"
    reason: str | None = None
    blurred: bool = True
    identity: str | None = None


@dataclass
class PipelineResult:
    image: np.ndarray
    faces: list
    instance_map: np.ndarray
    overlay_boxes: np.ndarray
    overlay_masks: np.ndarray
    result_image: np.ndarray
    timing_ms: dict
    meta: dict
```

`Face.crop_box` có thể nằm ngoài biên ảnh. `mask_box` mô tả vùng local được đặt lên ảnh gốc.

### 4.3 Luồng xử lý từng bước

| Bước | Module | Input | Output |
|---|---|---|---|
| 0. Tiền xử lý | `preprocess_input.py` | Bytes, đường dẫn hoặc file-like | RGB uint8 và metadata |
| 1. Phát hiện | `retinaface.py` | Ảnh RGB, ngưỡng confidence | `Detection[]` |
| 2. Crop | `crop.py` | Ảnh và box | Crop 256×256 và `crop_box` |
| 3. Phân vùng | `unet.py` | Batch crop float32 | Prob map và mask nhị phân |
| 4. Ghép mask | `mask_mapping.py` | Mask local và `crop_box` | `mask_local`, `instance_map` |
| 5. Che mặt | `privacy_filter.py` | Ảnh và danh sách ID chọn | Ảnh đã xử lý |
| 6. Hiển thị và export | `visualize.py`, `app.py` | Ảnh và kết quả mặt | Overlay PNG, mask PNG, JSON |

### 4.4 Quy tắc phát hiện và lọc

- Box bị clip trong biên ảnh.
- Các box có kích thước quá nhỏ bị loại nếu không hợp lệ.
- Detection được lọc NMS với `nms_iou_threshold = 0.38` trong logic pipeline.
- Giới hạn tối đa `max_faces = 200`.
- Các mặt được sắp theo tọa độ `x1` để ID ổn định.
- Nhiều box gần nhau có thể bị loại để tránh trùng phát hiện.

### 4.5 Trường hợp dự phòng

| Tình huống | Hành vi hiện tại | Trạng thái báo cáo |
|---|---|---|
| Không có mặt | Trả `faces=[]`, hiển thị thông báo và nút tải ảnh gốc | `num_faces = 0` |
| Mặt nhỏ hơn `min_face_px` | Không chạy U-Net; dùng ellipse nội tiếp bbox | `fallback_ellipse`, `too_small` |
| Mask rỗng hoặc tỷ lệ mask/bbox dưới 30% | Dùng ellipse nội tiếp bbox | `fallback_ellipse`, `bad_mask` |
| Mặt sát biên | Crop có phần vượt biên, sau đó clip khi ghép mask | `ok` hoặc fallback tùy mask |
| Nhiều mặt chồng nhau | Ghép theo instance map và id; ưu tiên ID được tạo | `ok` |
| Ảnh lớn | Thu nhỏ về cạnh dài tối đa `max_side` | Meta có `scale` |
| Thiếu checkpoint U-Net | Chặn pipeline và báo thiếu `models/unet_best.pt` | Hiển thị `FileNotFoundError` |
| Ảnh hỏng / sai định dạng / quá 10 MB | Báo lỗi thân thiện, không crash app | UI hiển thị thông báo |

---

## 5. Cấu hình inference

Cấu hình chính tại `configs/pipeline.yaml`; giá trị mặc định tại `src/utils/config.py`.

```yaml
device: auto
preprocess:
  max_side: 4096
  max_file_mb: 10
detection:
  model: retinaface_resnet50
  conf_threshold: 0.4
  max_faces: 200
crop:
  margin: 0.25
  size: 256
  min_face_px: 20
segmentation:
  weights: models/unet_best.pt
  mask_threshold: 0.5
  min_mask_ratio: 0.30
  batch_size: 16
privacy:
  mode: blur
  strength: 6
  region: mask
  dilate_px: 4
  feather_px: 3
  solid_color: [30, 30, 30]
```

### Ý nghĩa tham số quan trọng

- `conf_threshold`: ngưỡng điểm tin cậy của RetinaFace.
- `max_faces`: giới hạn số mặt chấp nhận.
- `margin`: mở rộng vùng crop quanh box.
- `size`: kích thước crop đưa vào U-Net.
- `min_face_px`: ngưỡng fallback ellipse.
- `mask_threshold`: ngưỡng xác suất tạo mask nhị phân.
- `min_mask_ratio`: tỷ lệ tối thiểu mask cắt trùng bbox.
- `device=auto`: ưu tiên CUDA, fallback CPU.

Ngưỡng mặc định ưu tiên tìm thêm khuôn mặt. Trong app, người dùng có thể hạ/ tăng `Ngưỡng phát hiện` bằng thanh bên.

---

## 6. API nội bộ

### 6.1 Phân tích

```python
result = pipeline.analyze(
    image,
    conf_threshold=0.4,
    source={"type": "image", "filename": "anh.jpg"},
    mask_threshold=0.5,
)
```

`PipelineResult` chứa ảnh và dữ liệu dự đoán. Hàm không thực hiện che mặt.

### 6.2 Che khuôn mặt

```python
result_image = pipeline.apply_privacy(
    result,
    selected_ids=[1, 2],
    mode="blur",
    strength=6,
    region="mask",
)
```

Các mode hiện có: `blur`, `pixelate`, `solid`. Các region hiện có: `mask`, `bbox`, `ellipse`.

### 6.3 Báo cáo JSON

Hàm `pipeline.report(result)` trả dữ liệu có thể serialize trực tiếp bằng `json.dumps()`.

Các nhóm trường chính:

```json
{
  "source": {"type": "image", "filename": "sample.jpg"},
  "image": {"width": 1920, "height": 1080},
  "num_faces": 2,
  "faces": [],
  "timing_ms": {},
  "device": "cuda",
  "params": {}
}
```

Mỗi phần tử trong `faces` có ít nhất:

- `id`
- `bbox`
- `size_px`
- `score`
- `status`
- `reason`
- `blurred`
- `identity`
- `mask_area_px`

Thời gian đo theo `timing_ms` và có thể thay đổi theo cấu hình/thời gian chạy.

---

## 7. Thiết kế giao diện Streamlit

### 7.1 Thanh bên

- Tiêu đề **Tùy chọn**.
- Slider **Ngưỡng phát hiện**: `0.10` đến `0.90`, mặc định `0.40`.
- Khi privacy bật: chọn kiểu che, độ mạnh và vùng che.
- Ghi chú xử lý trong phiên, không lưu ảnh.

### 7.2 Tab ảnh

Luồng:

```text
Tải ảnh hoặc chọn ảnh mẫu
  -> chọn ngưỡng
  -> bấm Phân tích
  -> xem kết quả
  -> tải PNG / JSON
```

Sau khi phân tích, ứng dụng hiển thị:

- Ba metric: số khuôn mặt, số mặt phân vùng bằng U-Net và thời gian xử lý.
- Tabs: **Ảnh gốc**, **Phát hiện**, **Phân vùng**.
- Thẻ khuôn mặt: thumbnail, ID, score, kích thước và fallback status.
- Nút tải ảnh phân vùng.
- Nút tải báo cáo JSON.

### 7.3 Tab khung hình video

Luồng:

```text
Tải video
  -> xem metadata
  -> chọn chỉ số frame
  -> xem khung hình
  -> bấm Phân tích khung hình này
```

Đầu ra hiện tại là kết quả của đúng một frame; app không tạo hoặc tải xuống video mới.

### 7.4 Tab so sánh

Tab này chỉ xuất hiện khi `ENABLE_PRIVACY = True`. Nó so sánh vùng che `bbox` và `mask`, đồng thời tính tỷ lệ phần trăm diện tích ảnh bị che.

---

## 8. Cấu trúc thư mục thực tế

```text
face_detection_segmentation/
├── app.py
├── configs/
│   └── pipeline.yaml
├── docs/
│   ├── PRODUCT_AND_SYSTEM_DESIGN.md
│   ├── thiet_ke_face_detection_segmentation.md
│   └── ...
├── src/
│   ├── detection/
│   │   └── retinaface.py
│   ├── segmentation/
│   │   └── unet.py
│   ├── pipeline/
│   │   ├── crop.py
│   │   ├── inference.py
│   │   ├── mask_mapping.py
│   │   ├── preprocess_input.py
│   │   ├── privacy_filter.py
│   │   ├── schemas.py
│   │   └── visualize.py
│   ├── utils/
│   │   ├── config.py
│   │   └── io.py
│   ├── evaluate_wider_face.py
│   ├── run_demo.py
│   └── demo_segment.py
├── tests/
│   └── test_pipeline_logic.py
├── models/
├── outputs/
├── requirements.txt
├── README.md
├── celebamask_unet_data_model_evaluation.ipynb
└── wider_face_data_model_evaluation1.ipynb
```

`models/unet_best.pt` là checkpoint cần thiết cho đường phân vùng U-Net. Chương trình kiểm tra checkpoint trước khi nạp pipeline.

---

## 9. Kiểm thử

Kiểm thử logic nằm tại `tests/test_pipeline_logic.py` và dùng detector/segmenter giả lập. Các ca test bao gồm:

- Không phát hiện mặt.
- Đọc ảnh hợp lệ/hỏng/quá dung lượng.
- Sắp xếp ID theo vị trí ngang.
- Mặt nhỏ fallback ellipse.
- Mask rỗng fallback ellipse.
- Mặt sát biên.
- Chỉ thay pixel của mặt được chọn.
- Ba mode privacy.
- Ba region privacy.
- Chồng lấn mask.
- Serialize báo cáo JSON.
- Thu nhỏ ảnh lớn.
- Lọc theo ngưỡng confidence.
- Làm sạch mask.

Chạy:

```powershell
python -m pytest tests -q
```

Hoặc:

```powershell
python tests/test_pipeline_logic.py
```

---

## 10. Evaluation

### Detection

Lệnh đánh giá hiện tại:

```powershell
python src/evaluate_wider_face.py --limit 200
```

Báo cáo đo:

- Precision
- Recall
- True Positive
- False Positive
- False Negative

Kết quả có thể đọc theo seed và giới hạn số ảnh để đảm bảo thử nghiệm lặp lại.

### Segmentation

Notebook `celebamask_unet_data_model_evaluation.ipynb` hỗ trợ huấn luyện và đánh giá U-Net trên dữ liệu CelebAMask-HQ.

Metric phù hợp:

- IoU
- Dice
- Pixel accuracy
- Recall theo kích thước vùng mặt
- Phân tích mặt quá nhỏ và mặt che khuất

---

## 11. Rủi ro và giới hạn

| Rủi ro | Tác động | Biện pháp hiện có |
|---|---|---|
| Mặt nhỏ hoặc che khuất | Mask không chính xác | Ellipse fallback và ngưỡng mặt nhỏ |
| Detection bỏ sót | Mặt không được xử lý | Ngưỡng mặc định thấp; người dùng có thể điều chỉnh |
| Detection nhận nhầm | Mask che thừa nền | NMS, giới hạn số mặt, kiểm tra thủ công |
| Domain gap | Model kém trên ảnh đám đông | Crop, margin, fallback và đánh giá theo kích thước |
| Thiếu checkpoint | Không chạy U-Net | Kiểm tra file và thông báo thiếu trọng số |
| GPU không khả dụng | Chạy chậm | Tự chuyển CPU |
| Ảnh lớn | Tốn RAM/GPU | Giới hạn cạnh dài theo `max_side` |
| Privacy chưa được đảm bảo tuyệt đối | Blur/che mặt có thể không ổn nếu phát hiện sai | Ghi rõ cần kiểm tra thủ công |

Blur nhẹ không thay thế ẩn danh pháp lý. Người dùng cần xem lại toàn bộ kết quả trước khi chia sẻ.

---

## 12. Hướng phát triển đề xuất

### Ưu tiên P0

- Chuẩn bị và kiểm tra checkpoint U-Net thực tế.
- Chạy full pipeline trên CPU và CUDA.
- Thêm test cho mask mapping với nhiều mặt và mặt sát biên.
- Hoàn thiện trạng thái lỗi checkpoint trong Streamlit.
- Cập nhật ảnh mẫu nếu thư mục `samples/` được bổ sung.

### Ưu tiên P1

- Bật và hoàn thiện tab che mặt trong giao diện.
- Thêm lựa chọn `segmented control` thay radio nếu phù hợp phiên bản Streamlit.
- Hỗ trợ tải ảnh đã che dạng PNG.
- Xuất video đã xử lý từng frame.
- Bổ sung metric đánh giá theo kích thước mặt.

### Ưu tiên P2

- Webcam trực tiếp.
- Theo dõi khuôn mặt qua frame.
- Nhận diện danh tính chỉ khi có yêu cầu riêng và cơ sở pháp lý.
- Tối ưu batch và caching.

Mọi thay đổi về phạm vi model hoặc UI phải cập nhật đồng bộ tài liệu, cấu hình, test và README.
