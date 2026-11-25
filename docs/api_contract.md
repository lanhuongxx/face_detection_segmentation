# API Contract — Face Detection & Segmentation Integration (Team 5)

Tài liệu này quy định **giao diện chuẩn (API contract)** giữa các module trong hệ thống phát hiện, phân vùng và che mờ khuôn mặt (**RetinaFace → Crop → U-Net → Mask Map-back → Privacy Filter / Video / Streamlit**).

---

## 1. Quy ước dữ liệu chung (Data Conventions)

| Thành phần | Kiểu dữ liệu | Shape | Giá trị / Hệ màu | Ghi chú |
|---|---|---|---|---|
| **Input / Output Image** | `np.ndarray` (`uint8`) | `(H, W, 3)` | RGB, `[0, 255]` | Nội bộ pipeline luôn dùng **RGB**. Chuyển BGR ↔ RGB chỉ thực hiện tại biên OpenCV (`cv2.imread`, `cv2.VideoCapture`, `cv2.VideoWriter`). |
| **Bounding Box (`bbox`)** | `tuple[int, int, int, int]` | 4 phần tử | `(x1, y1, x2, y2)` | Tọa độ **ảnh gốc**, đơn vị pixel: `0 <= x1 < x2 <= W`, `0 <= y1 < y2 <= H`. |
| **Confidence (`score`)** | `float` | scalar | `[0.0, 1.0]` | Độ tin cậy từ detector (RetinaFace / Haar). |
| **Landmarks** | `dict[str, tuple[float, float]] \| None` | 5 điểm (tùy chọn) | Tọa độ ảnh gốc `(x, y)` | Keys chuẩn: `left_eye`, `right_eye`, `nose`, `left_mouth`, `right_mouth`. |
| **Crop Probability Mask** | `np.ndarray` (`float32`) | `(target_H, target_W)` | `[0.0, 1.0]` | Output xác suất từ U-Net trên ảnh crop. |
| **Crop Binary Mask** | `np.ndarray` (`uint8`) | `(target_H, target_W)` | `{0, 1}` | Ngưỡng hóa từ `crop_prob_mask >= mask_threshold`. |
| **Full Probability Mask** | `np.ndarray` (`float32`) | `(orig_H, orig_W)` | `[0.0, 1.0]` | Mask sau khi map-back về đúng kích thước và tọa độ ảnh gốc. |
| **Full Binary Mask** | `np.ndarray` (`uint8`) | `(orig_H, orig_W)` | `{0, 1}` | Mask nhị phân trên tọa độ ảnh gốc. |

---

## 2. Cấu trúc dữ liệu chuẩn (`src/contracts.py`)

### 2.1 `FaceDetection` (Output của Detection Module)
```python
@dataclass(frozen=True)
class FaceDetection:
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) trên ảnh gốc
    score: float = 1.0
    landmarks: dict[str, tuple[float, float]] | None = None
```
- **Ràng buộc khi không có khuôn mặt:** Detector **phải** trả về danh sách rỗng `[]` (không trả `None` và không raise exception).

### 2.2 `CropMetadata` & `FaceCrop` (Output của Crop Module)
```python
@dataclass(frozen=True)
class CropMetadata:
    image_size: tuple[int, int]       # (orig_H, orig_W)
    orig_bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) bbox phát hiện ban đầu
    crop_box: tuple[int, int, int, int]   # (cx1, cy1, cx2, cy2) vùng cắt thực tế (đã cộng margin & clamp)
    crop_size: tuple[int, int]        # (cy2 - cy1, cx2 - cx1) trước khi resize
    target_size: tuple[int, int]      # (target_H, target_W) đầu vào U-Net, VD: (256, 256)
    margin_ratio: float = 0.15

@dataclass
class FaceCrop:
    image: np.ndarray                 # uint8, (target_H, target_W, 3), RGB
    metadata: CropMetadata
    detection: FaceDetection
```
- **Quy tắc Crop:**
  1. Mở rộng `orig_bbox` thêm `margin_ratio` theo chiều rộng và chiều cao của bbox.
  2. Clamp `(cx1, cy1, cx2, cy2)` vào biên ảnh gốc `[0, orig_W] × [0, orig_H]`.
  3. Cắt vùng `image[cy1:cy2, cx1:cx2]` và resize về `target_size`.

### 2.3 `FaceSegmentationResult` & `PipelineOutput` (Output của `inference.py`)
```python
@dataclass
class FaceSegmentationResult:
    detection: FaceDetection
    crop_metadata: CropMetadata
    crop_prob_mask: np.ndarray        # float32, (target_H, target_W)
    crop_mask: np.ndarray             # uint8, (target_H, target_W), {0, 1}
    full_prob_mask: np.ndarray        # float32, (orig_H, orig_W)
    full_mask: np.ndarray             # uint8, (orig_H, orig_W), {0, 1}

@dataclass
class PipelineOutput:
    image_size: tuple[int, int]       # (orig_H, orig_W)
    faces: list[FaceSegmentationResult]
    combined_prob_mask: np.ndarray    # float32, (orig_H, orig_W)
    combined_mask: np.ndarray         # uint8, (orig_H, orig_W), {0, 1}
    timings_ms: dict[str, float]      # detect_ms, crop_ms, segment_ms, map_back_ms, total_ms
```
- **Quy tắc Map-back (đặc biệt khi có nhiều khuôn mặt):**
  1. Resize `crop_prob_mask` từ `(target_H, target_W)` về đúng `(crop_H, crop_W) = (cy2 - cy1, cx2 - cx1)` bằng nội suy bilinear.
  2. Đặt patch vào canvas toàn 0 kích thước `(orig_H, orig_W)` tại slice `[cy1:cy2, cx1:cx2]`.
  3. Khi ghép nhiều khuôn mặt vào `combined_prob_mask`, lấy `np.maximum` theo từng pixel để các khuôn mặt gần nhau hoặc chồng lấn không ghi đè mất nhau.
  4. Ngưỡng hóa `combined_mask = (combined_prob_mask >= mask_threshold).astype(np.uint8)`.

---

## 3. Protocols cho Detection & Segmentation (`src/contracts.py`)

### 3.1 `FaceDetectorProtocol` (`src/detection/retinaface.py`)
```python
class FaceDetectorProtocol(Protocol):
    def detect(self, image: np.ndarray) -> list[FaceDetection]:
        ...
```
Ngoài chuẩn `FaceDetection`, `RetinaFaceDetectorAdapter` trong `src/inference.py` còn hỗ trợ tự động chuẩn hóa các dạng output phổ biến từ module `retinaface.py`:
- `list[FaceDetection]`
- `list[dict]` với keys `bbox` / `box` / `facial_area`, `score` / `confidence`, `landmarks`
- `dict` kiểu thư viện `retina-face` (`{"face_1": {"facial_area": [...], "score": ..., "landmarks": ...}}`)

### 3.2 `FaceSegmentorProtocol` (`src/segmentation/unet.py`)
```python
class FaceSegmentorProtocol(Protocol):
    def predict(self, crop_image: np.ndarray) -> np.ndarray:
        """Nhận ảnh crop RGB uint8 (H, W, 3), trả về prob_mask float32 (H, W) trong [0, 1]."""
        ...
```
`UNetSegmentorAdapter` hỗ trợ:
- Bọc trực tiếp một đối tượng tuân theo `FaceSegmentorProtocol` hoặc callable.
- Bọc một PyTorch `nn.Module` / checkpoint `.pth` (tự xử lý chuẩn hóa tensor `(1, 3, H, W)` và sigmoid nếu output là logits).
- Fallback sang heuristic mask nhẹ khi chưa có file trọng số U-Net (để demo/test pipeline chạy end-to-end).

---

## 4. Contract của Privacy Filter, Video & App

### 4.1 `PrivacyFilter` (`src/privacy_filter.py`)
- Chế độ hỗ trợ (`PrivacyMode`):
  - `"blur"`: Làm mờ Gaussian **chỉ theo vùng pixel của mask** (có feathering biên mượt).
  - `"pixelate"`: Làm mờ dạng mosaic ô vuông theo mask.
  - `"mask"`: Phủ màu bán trong suốt lên vùng mask (kèm bbox tùy chọn).
  - `"mask_only"`: Hiển thị mask nhị phân/màu trên nền đen.
  - `"blur_with_bbox"`: Làm mờ theo mask + vẽ bbox và confidence score.
  - `"original"`: Trả về bản sao ảnh gốc.
- Lưu ảnh: `save_image(image, output_path, quality=95)` và `encode_image(image, ext=".png")`.

### 4.2 `VideoPrivacyProcessor` (`src/video_pipeline.py`)
- Input: đường dẫn video gốc + `Pipeline` + `PrivacyFilterConfig` + `VideoConfig`.
- Đọc `input_fps = cap.get(cv2.CAP_PROP_FPS)` và **giữ nguyên `output_fps == input_fps`** khi khởi tạo `cv2.VideoWriter`.
- Nếu cấu hình `frame_stride > 1`, vẫn ghi đủ số frame gốc (tái sử dụng mask của frame trước đó) để **tốc độ phát và thời lượng video không bị sai lệch**.
