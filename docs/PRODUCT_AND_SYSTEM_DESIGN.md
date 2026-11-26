# Face Detection & Segmentation: Thiết kế sản phẩm và hệ thống Inference

> **Nhóm 5 · Đề tài:** Phát hiện và phân vùng khuôn mặt trong ảnh đám đông bằng RetinaFace + U-Net

> **Mục đích file:** (1) sản phẩm inference hướng tới là gì, có giao diện nào; (2) design system "nhận gì, trả gì" 
> **Quy ước:** các con số ngưỡng (threshold, kích thước, ...) là **giá trị đề xuất ban đầu**, sẽ chỉnh sau khi chạy thử và benchmark. Không phải kết quả đã đạt.

---

## 0. Tóm tắt quyết định (đọc phần này trước)

| Câu hỏi | Quyết định đề xuất |
|---|---|
| Sản phẩm là gì? | **FaceShield (tên tạm): công cụ che mặt (privacy blur) cho ảnh đông người.** Người dùng upload ảnh, hệ thống tìm từng khuôn mặt, tách đúng vùng mặt, che mờ theo hình dạng khuôn mặt rồi cho tải ảnh về. |
| Bounding box là gì trong sản phẩm? | Bounding box **chỉ là bước trung gian** (hướng modeling). Sản phẩm trả ra **ảnh đã che mặt + báo cáo từng khuôn mặt**. |
| Vì sao cần segmentation? | Làm mờ theo mask che đúng khuôn mặt, ít phá nền hơn che nguyên hình chữ nhật. UI có tab **So sánh Box vs Mask** để chứng minh điều này. |
| Giao diện? | Web app **Gradio** một trang: upload → chọn chế độ → xem 4 tab (Gốc / Phát hiện / Phân vùng / Kết quả) → chọn khuôn mặt cần che → tải về. Wireframe ở mục 4. |
| Design system? | Mỗi module có **hợp đồng dữ liệu** rõ: kiểu dữ liệu, hệ tọa độ, shape, tham số, trường hợp biên, JSON đầu ra. Xem mục 3. |

**Một điểm cần chốt cả nhóm:** đề tài gốc ghi "camera an ninh, nhận diện khuôn mặt trong đám đông", nhưng hệ thống thực tế **không nhận diện danh tính**, chỉ phát hiện và phân vùng. Nên chốt use case chính là **che mặt trước khi chia sẻ ảnh** (khớp sheet Document, task `privacy_filter.py`). Camera an ninh/webcam giữ ở mức mở rộng P2.

---

## 1. Nhận xét của cô → cách xử lý

| # | Nhận xét | Vấn đề thực sự | Cách xử lý trong tài liệu này |
|---|---|---|---|
| 1 | "Mục tiêu xây dựng hệ thống inference thôi" | Nhóm đang nghĩ như bài ML (train, đánh giá), chưa có "người dùng" của hệ thống | Mục 2: xác định người dùng, user story, phạm vi |
| 2 | "Nếu inference thì sản phẩm hướng tới là gì?" | Chưa có sản phẩm cụ thể | Sản phẩm = công cụ che mặt ảnh đông người (mục 2.1) |
| 3 | "Chọn inference thì sản phẩm phải có giao diện" | Backlog có `app.py` nhưng chưa thiết kế UI | Mục 4: user flow, wireframe, ánh xạ UI ↔ module |
| 4 | "Output bounding box chỉ là hướng modeling, full pipeline sẽ khác" | Sheet Scope đang ghi "trả về bounding box" | Sửa Scope: output cuối là **ảnh đã xử lý + mask + báo cáo JSON** (mục 3.5) |
| 5 | "Design system nhận gì trả gì còn chung chung" | Bảng module hiện chỉ có 1 dòng mô tả mỗi ô | Mục 3: hợp đồng dữ liệu đầy đủ cho 7 bước |

---

## 2. Định nghĩa sản phẩm

### 2.1 Product statement

> **FaceShield** là ứng dụng web cho phép người dùng upload ảnh có nhiều người, tự động phát hiện và phân vùng từng khuôn mặt, che mờ các khuôn mặt được chọn theo đúng hình dạng khuôn mặt, và tải ảnh đã che về.

**Giá trị cho người dùng:** không phải khoanh tay từng mặt trong ảnh 30 người. Thao tác chính: *upload → kiểm tra → tải về*.

### 2.2 Người dùng mục tiêu

| Persona | Nhu cầu | Ví dụ |
|---|---|---|
| Người phụ trách truyền thông (fanpage trường, CLB, sự kiện) | Đăng ảnh sự kiện nhưng che mặt người không đồng ý | Ảnh hội trại, ảnh khai giảng |
| Giáo viên / nhà trường | Chia sẻ ảnh lớp học, không lộ mặt học sinh | Ảnh hoạt động ngoại khóa |
| Phóng viên / người làm tài liệu | Che mặt người qua đường trong ảnh đường phố | Ảnh phóng sự |

### 2.3 User stories và tiêu chí chấp nhận

| ID | User story | Tiêu chí chấp nhận (kiểm thử được) | Ưu tiên | Module chính |
|---|---|---|---|---|
| US1 | Là người dùng, tôi upload 1 ảnh và nhận lại ảnh đã che toàn bộ mặt phát hiện được | Upload jpg/png → có ảnh kết quả; không crash với ảnh không có mặt | P0 | Toàn pipeline |
| US2 | Tôi xem hệ thống phát hiện được những khuôn mặt nào | Tab "Phát hiện" vẽ box + số thứ tự + score | P0 | Detection |
| US3 | Tôi xem vùng mặt được tách ra thế nào | Tab "Phân vùng" phủ màu mask từng mặt; mask đặt đúng vị trí khi có nhiều mặt | P0 | Segmentation, Mask mapping |
| US4 | Tôi chọn che hay không che từng khuôn mặt | Danh sách mặt có checkbox; bỏ chọn thì mặt đó không bị che | P0 | Privacy filter, UI |
| US5 | Tôi chọn kiểu che (làm mờ / pixel / tô màu) và độ mạnh | 3 chế độ + slider; kết quả đổi theo | P0 | Privacy filter |
| US6 | Tôi tải ảnh kết quả và báo cáo | Nút tải PNG/JPG và JSON | P0 | Export |
| US7 | Tôi so sánh che theo box và che theo mask | Tab "So sánh" hiển thị 2 ảnh cạnh nhau | P1 | Privacy filter |
| US8 | Tôi xử lý video ngắn | Upload mp4, xuất mp4 giữ FPS gốc | P1 | Video pipeline |
| US9 | Tôi dùng webcam gần thời gian thực | Hiển thị FPS và nêu giới hạn | P2 | Stretch |

### 2.4 Phạm vi

**Trong phạm vi (In scope)**
- Ảnh tĩnh có 0, 1 hoặc nhiều khuôn mặt.
- Phát hiện (RetinaFace) → cắt → phân vùng (U-Net) → ghép mask về ảnh gốc → che mặt → hiển thị, xuất kết quả.
- Báo cáo từng khuôn mặt và thời gian xử lý từng bước.
- Video ngắn (P1).

**Ngoài phạm vi (Out of scope)**
- Nhận diện danh tính (không biết "người này là ai").
- Theo dõi (tracking) một người qua các khung hình.
- Che mặt không thể khôi phục ở mức bảo mật pháp lý. Đây là công cụ hỗ trợ, người dùng vẫn phải kiểm tra kết quả (xem mục 7).
- Triển khai production, đăng nhập, lưu trữ ảnh người dùng.

### 2.5 Vì sao chọn metric theo hướng "không bỏ sót mặt"

Với công cụ che mặt, **bỏ sót một khuôn mặt (False Negative) là lỗi nặng nhất**, vì mặt đó lộ ra. Phát hiện nhầm (False Positive) chỉ làm mờ thừa một vùng. Do đó:
- **Recall** của detection quan trọng hơn Precision, nên ngưỡng `conf_threshold` mặc định đặt thấp, người dùng có thể bỏ chọn các box thừa trong UI.
- Báo cáo đánh giá nên nêu rõ Recall theo kích thước mặt (nhỏ / vừa / lớn), vì mặt nhỏ trong đám đông là nơi dễ bỏ sót nhất.

Đây cũng là câu trả lời tốt nếu cô hỏi "vì sao chọn các metric này".

---

## 3. Thiết kế hệ thống (Design System)

### 3.1 Kiến trúc tổng thể

```
        Ảnh (jpg / png / webp)
                │
                ▼
┌───────────────────────────────┐
│ 0. Preprocess                 │  sửa EXIF, về RGB, giới hạn kích thước
└───────────────┬───────────────┘
                ▼
┌───────────────────────────────┐
│ 1. Detection  (RetinaFace)    │  → danh sách Face: bbox, score, landmarks
└───────────────┬───────────────┘
                ▼
┌───────────────────────────────┐
│ 2. Crop                       │  → crop vuông 256×256 + crop_box (tọa độ gốc)
└───────────────┬───────────────┘
                ▼
┌───────────────────────────────┐
│ 3. Segmentation  (U-Net)      │  → mask xác suất → mask nhị phân mỗi crop
└───────────────┬───────────────┘
                ▼
┌───────────────────────────────┐
│ 4. Mask mapping               │  → mask về đúng tọa độ ảnh gốc
└───────────────┬───────────────┘
                ▼
┌───────────────────────────────┐
│ 5. Privacy filter             │  → ảnh đã che (blur / pixel / solid)
└───────────────┬───────────────┘
                ▼
┌───────────────────────────────┐
│ 6. Visualization & Export     │  → ảnh overlay, ảnh kết quả, JSON
└───────────────────────────────┘
                │
                ▼
          Giao diện Gradio
```

**Hai luồng riêng biệt (giữ như sheet Document):**
- *Luồng fine-tune (offline):* ảnh mặt + mask chuẩn (CelebAMask-HQ) → U-Net dự đoán → so với mask chuẩn, cập nhật → lưu `models/unet_best.pt`.
- *Luồng inference (demo):* pipeline 0→6 ở trên, **không train**, chỉ nạp weights.

### 3.2 Quy ước dùng chung cho mọi module

| Hạng mục | Quy ước |
|---|---|
| Ảnh trong bộ nhớ | `numpy.ndarray`, shape `(H, W, 3)`, `uint8`, **RGB** (không phải BGR). Nhớ đổi khi dùng OpenCV |
| Hệ tọa độ | Pixel của **ảnh gốc sau preprocess**, gốc (0,0) ở góc trên trái |
| Bounding box | `(x1, y1, x2, y2)`, kiểu `int`, `x2 > x1`, `y2 > y1`, đã clip trong ảnh |
| Mask nhị phân | `numpy.ndarray`, `uint8`, giá trị `{0, 1}` |
| Mask xác suất | `float32`, giá trị `[0, 1]` |
| ID khuôn mặt | `int` bắt đầu từ 1, sắp xếp theo `x1` tăng dần (ổn định giữa các lần chạy) |
| Thiết bị | `cuda` nếu có, ngược lại `cpu`, đọc từ config |
| Lỗi | Module ném exception có tên rõ; pipeline bắt và trả thông báo cho UI (mục 3.7) |

> **Vì sao lưu mask theo `crop_box` (local) thay vì mask cỡ ảnh gốc cho mỗi mặt?** Ảnh 4000×3000 với 40 mặt sẽ cần 40 mask lớn, rất tốn RAM. Mỗi mặt chỉ giữ mask nhỏ + `crop_box`; mask toàn ảnh chỉ ghép **một lần** ở bước 4.

### 3.3 Hợp đồng dữ liệu từng module

#### Module 0: Preprocess
| Mục | Nội dung |
|---|---|
| **Input** | File ảnh (bytes/đường dẫn) |
| **Làm gì** | Kiểm tra định dạng, đọc ảnh, xoay theo EXIF, đổi về RGB. Nếu cạnh dài > `max_side` (4096) thì thu nhỏ giữ tỉ lệ và lưu `scale` |
| **Output** | `image` `(H,W,3)` RGB uint8, `meta = {filename, orig_size, work_size, scale}` |
| **Lỗi** | `UnsupportedFormat`, `CorruptImage`, `FileTooLarge` (> 10 MB) |

#### Module 1: Detection (RetinaFace)
| Mục | Nội dung |
|---|---|
| **Input** | `image` RGB; tham số `conf_threshold` (0.5), `nms_iou` (0.4) |
| **Làm gì** | Chạy RetinaFace pretrained, lọc theo score, NMS |
| **Output** | `List[Detection]`: `bbox`, `score` (float 0–1), `landmarks` `(5,2)` hoặc `None` |
| **Trường hợp biên** | Không có mặt → **danh sách rỗng** (không phải lỗi). Box vượt biên ảnh → clip |
| **Backlog** | `src/detection/retinaface.py`; baseline Haar cùng interface để so sánh |

> Haar baseline phải trả **cùng kiểu `List[Detection]`** (score có thể đặt 1.0) để `detect_metrics.py` chạy chung một đường.

#### Module 2: Crop
| Mục | Nội dung |
|---|---|
| **Input** | `image`, `List[Detection]`; tham số `margin` (0.25), `crop_size` (256) |
| **Làm gì** | Với mỗi bbox: mở rộng thành **hình vuông** quanh tâm box (cạnh = `max(w,h)·(1+2·margin)`), cắt, **pad đen nếu vượt biên**, resize về `crop_size×crop_size` |
| **Output** | `List[CropItem]`: `face_id`, `crop` `(256,256,3)`, `crop_box` `(x1,y1,x2,y2)` **tọa độ ảnh gốc, có thể vượt biên ảnh** |
| **Vì sao vuông + margin** | U-Net train trên ảnh mặt gần vuông; margin giữ cằm, trán, tai và bù sai số của box |
| **Giới hạn** | Mặt có cạnh nhỏ hơn `min_face_px` (20 px) bị upscale rất mạnh, chất lượng mask kém. Xử lý ở mục 3.4 |

#### Module 3: Segmentation (U-Net)
| Mục | Nội dung |
|---|---|
| **Input** | Batch crop `(N,3,256,256)` float32, chuẩn hóa như lúc train; `mask_threshold` (0.5) |
| **Làm gì** | Forward U-Net → sigmoid → ngưỡng |
| **Output** | `prob` `(N,256,256)` float32 và `mask` `(N,256,256)` uint8 |
| **Định nghĩa "mặt" (cần chốt)** | Gộp các lớp CelebAMask-HQ: `skin, nose, l_eye, r_eye, l_brow, r_brow, mouth, u_lip, l_lip` (+ `l_ear, r_ear` tùy nhóm). **Loại**: hair, hat, cloth, neck, eyeglass. Ghi vào Data card và dùng đúng một định nghĩa ở cả train, đánh giá, demo |
| **Backlog** | `src/segmentation/unet.py`, `predict.py`; trọng số ở `models/unet_best.pt` |

#### Module 4: Mask mapping
| Mục | Nội dung |
|---|---|
| **Input** | `prob` mỗi mặt, `crop_box`, kích thước ảnh gốc |
| **Làm gì** | Resize **prob** (không phải mask nhị phân) về kích thước `crop_box` bằng nội suy tuyến tính, rồi mới ngưỡng, để biên mượt hơn. Cắt phần vượt biên, đặt vào canvas ảnh gốc |
| **Output** | `instance_map` `(H,W)` `uint16`: `0` = nền, `k` = mặt có `id = k`. Mỗi mặt cũng giữ `mask_local` theo `crop_box` |
| **Xử lý chồng lấn** | Hai mặt chồng nhau: pixel thuộc mặt có `score` cao hơn; vẫn có `mask_local` riêng để che đủ |
| **Kiểm thử** | Test đặt mask khi có ≥ 3 mặt, mặt sát biên ảnh, mặt bị pad |

#### Module 5: Privacy filter
| Mục | Nội dung |
|---|---|
| **Input** | `image`, `instance_map`, `selected_ids`, `mode` ∈ {`blur`, `pixelate`, `solid`}, `strength` (1–10), `region` ∈ {`mask`, `bbox`, `ellipse`}, `feather` (px) |
| **Làm gì** | Dilate mask 3–5 px (an toàn viền), làm mờ cạnh (feather), áp hiệu ứng **chỉ trong vùng chọn**, trộn với ảnh gốc |
| **Output** | `result_image` `(H,W,3)` RGB uint8 |
| **Quy tắc** | Mặt không có trong `selected_ids` thì giữ nguyên. `region=bbox` dùng cho tab so sánh |
| **Backlog** | `src/pipeline/privacy_filter.py` |

#### Module 6: Visualization & Export
| Mục | Nội dung |
|---|---|
| **Input** | `image`, `List[Face]`, `result_image`, `timing` |
| **Làm gì** | Vẽ 4 bản hiển thị (gốc, box, mask phủ màu, kết quả); xuất file |
| **Output** | `overlay_boxes`, `overlay_masks`, `result_image`, `report.json`, (tùy chọn) ZIP |
| **Quy ước vẽ** | Mỗi mặt một màu cố định theo `id`; nhãn `#id · score` |

### 3.4 Trường hợp biên và quy tắc dự phòng (fallback)

Phần này quan trọng với công cụ che mặt: **thà che thừa còn hơn để lộ mặt**.

| Tình huống | Quy tắc | `status` ghi trong báo cáo |
|---|---|---|
| Không phát hiện mặt nào | Trả ảnh gốc, `faces=[]`, UI báo "Không tìm thấy khuôn mặt" kèm gợi ý hạ ngưỡng | – |
| Mặt quá nhỏ (`min(w,h) < 20 px`) | Bỏ qua U-Net, dùng **ellipse nội tiếp trong bbox** để che | `fallback_ellipse` (lý do: `too_small`) |
| Mask bất thường (diện tích mask ∩ bbox < 30% bbox, hoặc mask rỗng) | Dùng ellipse trong bbox | `fallback_ellipse` (lý do: `bad_mask`) |
| Mặt sát biên ảnh | Pad khi cắt, cắt phần vượt khi ghép | `ok` |
| Nhiều mặt chồng nhau | Xem mục Mask mapping | `ok` |
| Mặt bị che một phần (khẩu trang, tay) | U-Net chỉ tách phần thấy được; chấp nhận, ghi vào giới hạn | `ok` |
| Ảnh rất lớn | Thu nhỏ về `max_side`, ghép kết quả lại theo `scale` | – |
| Mất GPU / hết VRAM | Tự chuyển CPU, cảnh báo chậm | – |
| Ảnh hỏng / sai định dạng | Thông báo lỗi thân thiện, không crash | – |

> Các ngưỡng 20 px và 30% là điểm khởi đầu. Task benchmark/đánh giá nên chỉnh chúng dựa trên tập val.

### 3.5 Cấu trúc đầu ra (JSON báo cáo)

Đây là **output thật của hệ thống**, thay cho "chỉ trả bounding box":

```json
{
  "image": { "filename": "su_kien.jpg", "width": 1920, "height": 1080 },
  "params": {
    "conf_threshold": 0.5, "margin": 0.25, "crop_size": 256,
    "mask_threshold": 0.5, "mode": "blur", "strength": 6, "region": "mask"
  },
  "num_faces": 2,
  "faces": [
    {
      "id": 1,
      "bbox": [412, 188, 470, 255],
      "score": 0.987,
      "landmarks": [[430,210],[455,209],[443,225],[433,240],[454,239]],
      "crop_box": [398, 170, 484, 256],
      "mask_area_px": 2874,
      "status": "ok",
      "reason": null,
      "blurred": true
    },
    {
      "id": 2,
      "bbox": [901, 340, 915, 357],
      "score": 0.612,
      "landmarks": null,
      "crop_box": null,
      "mask_area_px": 190,
      "status": "fallback_ellipse",
      "reason": "too_small",
      "blurred": true
    }
  ],
  "timing_ms": {
    "preprocess": 0, "detect": 0, "crop": 0,
    "segment": 0, "map_back": 0, "filter": 0, "total": 0
  },
  "device": "cuda"
}
```
*(các giá trị trên chỉ minh họa cấu trúc; `timing_ms` là số đo thực khi chạy).*

Ảnh không có mặt: `"num_faces": 0, "faces": []`.

### 3.6 Giao diện lập trình nội bộ (API contract: task của Nguyên)

```python
# src/pipeline/schemas.py
from dataclasses import dataclass
from typing import Optional, Literal
import numpy as np

BBox = tuple[int, int, int, int]  # x1, y1, x2, y2 (ảnh gốc)

@dataclass
class Face:
    id: int
    bbox: BBox
    score: float
    landmarks: Optional[np.ndarray] = None      # (5, 2)
    crop_box: Optional[BBox] = None             # có thể vượt biên ảnh
    mask_local: Optional[np.ndarray] = None     # (h, w) uint8 theo crop_box
    status: Literal["ok", "fallback_ellipse"] = "ok"
    reason: Optional[str] = None                # too_small | bad_mask
    blurred: bool = True

@dataclass
class PipelineResult:
    image: np.ndarray                           # RGB gốc
    faces: list[Face]
    instance_map: np.ndarray                    # (H, W) uint16
    overlay_boxes: np.ndarray
    overlay_masks: np.ndarray
    result_image: np.ndarray
    timing_ms: dict
    meta: dict

# src/pipeline/inference.py
class FacePipeline:
    def __init__(self, cfg_path: str = "configs/pipeline.yaml"): ...
    def analyze(self, image: np.ndarray) -> PipelineResult:
        """Bước 0–4 + 6: phát hiện + phân vùng, CHƯA che mặt."""
    def apply_privacy(self, result: PipelineResult, selected_ids: list[int],
                      mode: str, strength: int, region: str) -> np.ndarray:
        """Bước 5: che theo lựa chọn. Gọi lại rất nhanh khi người dùng đổi tùy chọn."""
```

> **Vì sao tách `analyze` và `apply_privacy`?** Người dùng đổi kiểu che hoặc bỏ chọn một mặt thì **không cần chạy lại RetinaFace/U-Net** (phần nặng nhất). UI sẽ mượt hơn nhiều và đo FPS/latency cũng rõ ràng hơn.

### 3.7 Cấu hình `configs/pipeline.yaml`

```yaml
device: auto            # auto | cuda | cpu
preprocess:
  max_side: 4096
  max_file_mb: 10
detection:
  model: retinaface
  weights: models/retinaface_resnet50.pth
  conf_threshold: 0.5   # thấp hơn nếu muốn ưu tiên Recall
  nms_iou: 0.4
crop:
  margin: 0.25
  size: 256
  min_face_px: 20
segmentation:
  weights: models/unet_best.pt
  mask_threshold: 0.5
  min_mask_ratio: 0.30
privacy:
  default_mode: blur
  default_strength: 6
  default_region: mask
  dilate_px: 4
  feather_px: 3
```

---

## 4. Thiết kế giao diện (UI)

### 4.1 Nguyên tắc
1. **Một màn hình, một việc chính:** upload → xem → tải. Không cần đăng nhập, không có menu phức tạp.
2. **Cho người dùng thấy hệ thống "nghĩ gì":** có tab Phát hiện / Phân vùng để kiểm tra trước khi tin kết quả (đồng thời là chỗ cho nhóm demo từng module).
3. **Người dùng có quyền sửa:** bỏ chọn mặt thừa, hạ ngưỡng nếu thiếu mặt.
4. **Luôn có kết quả hoặc thông báo rõ ràng**, không để màn hình trống hay lỗi khó hiểu.

### 4.2 Luồng người dùng

```
Mở app ─► Upload ảnh ─► [Phân tích] ─► Xem 4 tab ─► Chọn/bỏ chọn mặt
                                                │         │
                                                │         ▼
                                                │   Chọn kiểu che + độ mạnh
                                                ▼         │
                                         Không thấy mặt?  ▼
                                         hạ ngưỡng ◄── Tải ảnh / tải JSON
```

### 4.3 Màn hình chính (sau khi phân tích)

```
┌────────────────────────────────────────────────────────────────────────────┐
│  FaceShield · Che mặt trong ảnh đông người      [ Ảnh ] [ Video ] [ So sánh ]│
├──────────────────────────────────────────────┬─────────────────────────────┤
│  [ Gốc ] [ Phát hiện ] [ Phân vùng ] [Kết quả]│  1. Kiểu che                │
│ ┌──────────────────────────────────────────┐ │   (•) Làm mờ                │
│ │                                          │ │   ( ) Pixel hóa             │
│ │                                          │ │   ( ) Tô màu                │
│ │        VÙNG HIỂN THỊ ẢNH                 │ │                             │
│ │        (tab đang chọn)                   │ │  2. Độ mạnh                 │
│ │                                          │ │   [───────●─────]  6        │
│ │                                          │ │                             │
│ │                                          │ │  3. Vùng che                │
│ │                                          │ │   (•) Theo khuôn mặt (mask) │
│ └──────────────────────────────────────────┘ │   ( ) Hình chữ nhật (box)   │
│                                              │                             │
│  Đã tìm thấy 12 khuôn mặt · xử lý 1.4 s      │  4. Khuôn mặt cần che       │
│                                              │   [Chọn tất cả] [Bỏ chọn]   │
│  ┌──────────────────────────────────────┐    │   [x] #1  ▢  0.99  mask     │
│  │ Thời gian: detect ▇▇ segment ▇▇▇     │    │   [x] #2  ▢  0.97  mask     │
│  │ filter ▇  (chi tiết ở JSON)          │    │   [x] #3  ▢  0.61  ellipse  │
│  └──────────────────────────────────────┘    │   [ ] #4  ▢  0.52  mask     │
│                                              │   ...  (cuộn)               │
│                                              │                             │
│                                              │  Nâng cao ▸                 │
│                                              │   Ngưỡng phát hiện [──●───] │
│                                              │                             │
│                                              │  [ Tải ảnh ] [ Tải JSON ]   │
└──────────────────────────────────────────────┴─────────────────────────────┘
```

**Ý nghĩa các thành phần (và module đứng sau):**

| Thành phần UI | Hiển thị gì | Module phụ trách | Gọi hàm |
|---|---|---|---|
| Tab **Gốc** | Ảnh upload | Preprocess | – |
| Tab **Phát hiện** | Box + `#id · score` | Detection | `analyze().overlay_boxes` |
| Tab **Phân vùng** | Mask phủ màu từng mặt | Segmentation + Mask mapping | `analyze().overlay_masks` |
| Tab **Kết quả** | Ảnh đã che | Privacy filter | `apply_privacy()` |
| Danh sách khuôn mặt (checkbox) | Thumbnail, score, `status` | Face list từ `analyze()` | → `selected_ids` |
| Kiểu che / độ mạnh / vùng che | Tùy chọn | Privacy filter | `apply_privacy(mode, strength, region)` |
| Ngưỡng phát hiện (Nâng cao) | Slider | Detection | Chạy lại `analyze()` |
| Dòng thống kê | Số mặt, tổng thời gian | `timing_ms` | – |
| Nút tải | PNG/JPG, JSON | Export | – |

### 4.4 Màn hình khởi đầu (chưa upload)

```
┌────────────────────────────────────────────────────────────────────────────┐
│  FaceShield · Che mặt trong ảnh đông người      [ Ảnh ] [ Video ] [ So sánh ]│
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│        ┌────────────────────────────────────────────────────┐              │
│        │                                                    │              │
│        │         Kéo thả ảnh vào đây hoặc bấm để chọn       │              │
│        │            (JPG, PNG, WEBP · tối đa 10 MB)         │              │
│        │                                                    │              │
│        └────────────────────────────────────────────────────┘              │
│                                                                            │
│        Hoặc thử ảnh mẫu:   [ Mẫu 1 ]  [ Mẫu 2 ]  [ Mẫu 3 ]                │
│                                                                            │
│                                         [ Phân tích ]                      │
│                                                                            │
│   Lưu ý: ảnh chỉ xử lý trong phiên làm việc, không lưu lại.               │
└────────────────────────────────────────────────────────────────────────────┘
```

Ảnh mẫu (`samples/`) giúp demo cho cô mà không cần chuẩn bị ảnh, và cho thấy các tình huống khó (đông, nghiêng, tối). **Chỉ dùng ảnh được phép phân phối lại** (xem mục 7).

### 4.5 Tab "So sánh" (Box vs Mask) — P1

```
┌────────────────────────────────────────────────────────────────────────────┐
│  So sánh cách che                                                          │
│ ┌───────────────────────────────┐  ┌───────────────────────────────┐       │
│ │   Che theo BOX (hình chữ nhật)│  │   Che theo MASK (U-Net)       │       │
│ │                               │  │                               │       │
│ │   [ảnh]                       │  │   [ảnh]                       │       │
│ │                               │  │                               │       │
│ └───────────────────────────────┘  └───────────────────────────────┘       │
│  Diện tích bị che: 18.2% ảnh        Diện tích bị che: 11.7% ảnh            │
│  (số minh họa; tính thật từ mask)                                          │
└────────────────────────────────────────────────────────────────────────────┘
```

Tab này giải thích trực quan **vì sao cần segmentation**, và là chỗ trả lời câu "bounding box là đủ rồi, sao phải U-Net?". Có thể tính tỉ lệ diện tích bị che từ `instance_map` và bbox.

### 4.6 Trạng thái đặc biệt và thông báo

| Trạng thái | Thông báo hiển thị |
|---|---|
| Không có mặt | "Không tìm thấy khuôn mặt nào. Thử hạ **Ngưỡng phát hiện** trong mục Nâng cao." Hiện nút tải ảnh gốc |
| Đang xử lý | Spinner + "Đang phát hiện khuôn mặt..." rồi "Đang phân vùng..." |
| Mặt nhỏ dùng ellipse | Badge `ellipse` ở danh sách + chú thích: "Mặt quá nhỏ, che theo hình elip" |
| Sai định dạng | "Hệ thống hỗ trợ JPG, PNG, WEBP." |
| Ảnh quá lớn | "Ảnh vượt 10 MB. Hãy giảm kích thước rồi thử lại." |
| Chạy CPU | "Đang chạy bằng CPU nên có thể chậm." |

### 4.7 Tab "Video" (P1) và Webcam (P2)

- **Video (P1):** upload mp4 ngắn → chọn kiểu che (dùng cùng tùy chọn) → thanh tiến độ → xem/tải mp4. Xử lý **từng khung hình độc lập** (không tracking), nên mask có thể nhấp nháy; ghi rõ vào giới hạn. Giữ FPS gốc khi xuất.
- **Webcam (P2):** xem trực tiếp có che mặt, hiện FPS ở góc. Nêu rõ giới hạn tốc độ nếu chạy CPU.

### 4.8 Công nghệ giao diện

Đề xuất **Gradio Blocks**: có sẵn `Image`, `Tabs`, `CheckboxGroup`, `Radio`, `Slider`, `File`, `Video`; dễ chia `analyze()` và `apply_privacy()` thành hai callback; chạy được trên máy khác theo task README/setup. Streamlit cũng làm được, nhưng phải xử lý `session_state` nhiều hơn cho phần chọn mặt. **Chọn một và ghi vào README** (backlog đang ghi "Streamlit/Gradio").

---

## 5. Cấu trúc thư mục

Giữ cấu trúc nhóm đã dựng, **bổ sung** các phần in đậm để đủ cho pipeline và UI:

```
face_detection_segmentation/
│
├── data/                        # KHÔNG commit (gitignore)
│   ├── raw/                     #   WIDER FACE, CelebAMask-HQ gốc
│   ├── processed/               #   ảnh/mask đã tiền xử lý
│   └── splits/                  #   manifest train/val/test (commit file nhỏ)
│
├── samples/                     # ★ ảnh mẫu cho demo (chỉ ảnh được phép phân phối)
│
├── notebooks/
│   ├── 01_eda.ipynb             # Huy
│   ├── 02_unet_training.ipynb   # Hương
│   ├── 03_loss_comparison.ipynb # Hương
│   └── 04_benchmark.ipynb       # Việt
│
├── src/
│   ├── datasets/                # Huy: chuẩn bị dữ liệu
│   │   ├── preprocess.py
│   │   ├── face_dataset.py
│   │   ├── splits_data.py
│   │   └── make_binary_mask.py  #   CelebAMask-HQ → mask nhị phân
│   │
│   ├── detection/               # Uyên: RetinaFace
│   │   ├── retinaface.py
│   │   ├── haar_baseline.py
│   │   └── detect_metrics.py
│   │
│   ├── segmentation/            # Hương: U-Net
│   │   ├── unet.py
│   │   ├── losses.py
│   │   ├── train.py
│   │   └── predict.py
│   │
│   ├── pipeline/                # ★ Nguyên: ghép các module (hiện đang rải ở src/)
│   │   ├── schemas.py           #   Face, PipelineResult (mục 3.6)
│   │   ├── preprocess_input.py  #   Module 0 (EXIF, RGB, resize)
│   │   ├── crop.py              #   Module 2
│   │   ├── mask_mapping.py      #   Module 4
│   │   ├── inference.py         #   FacePipeline.analyze / apply_privacy
│   │   ├── privacy_filter.py    #   Module 5
│   │   ├── visualize.py         #   Module 6 (overlay, export JSON)
│   │   └── video_pipeline.py    #   P1
│   │
│   ├── evaluation/              # Yến
│   │   ├── seg_metrics.py
│   │   ├── evaluate.py
│   │   └── error_analysis.py
│   │
│   └── utils/
│       ├── config.py            # ★ đọc YAML
│       ├── io.py                # ★ đọc/ghi ảnh, video
│       └── benchmark.py         # Việt
│
├── configs/
│   ├── pipeline.yaml            # ★ cấu hình inference (mục 3.7)
│   ├── unet_train.yaml
│   └── eval.yaml
│
├── models/                      # KHÔNG commit weights (gitignore)
│   └── README.md                # ★ link tải + checksum weights
│
├── scripts/                     # ★
│   └── download_weights.py      #   tải weights để máy khác chạy lại demo
│
├── tests/                       # Việt
│   ├── test_detection.py
│   ├── test_segmentation.py
│   ├── test_pipeline.py
│   ├── test_mask_mapping.py     # ★ mặt sát biên, nhiều mặt, mặt bị pad
│   ├── test_privacy_filter.py   # ★ mặt không chọn phải giữ nguyên
│   └── fixtures/                # ★ ảnh test: không mặt / nhiều mặt / nghiêng / tối
│
├── docs/
│   ├── product_spec.md          # ★ chính là file này (rút gọn)
│   ├── data_card.md             # Huy: nguồn, giấy phép, giới hạn, định nghĩa "mặt"
│   ├── api_contract.md          # ★ mục 3
│   └── ui_wireframes.md         # ★ mục 4
│
├── outputs/                     # KHÔNG commit
│   ├── results/                 #   ảnh/JSON demo
│   └── logs/
│
├── app.py                       # Nguyên: Gradio, chỉ gọi src/pipeline
├── requirements.txt
├── .gitignore
└── README.md                    # cài đặt, tải weights, chạy demo
```

**Nguyên tắc phân tầng:**
- `app.py` **chỉ** gọi `FacePipeline`, không chứa logic xử lý ảnh.
- `src/pipeline/` gọi `detection/` và `segmentation/` qua interface ở mục 3.3, **không** import chéo nội bộ của hai module đó.
- `evaluation/` dùng lại `predict.py`/`retinaface.py`, không viết lại model.

> **Lưu ý đồng bộ:** Dashboard ghi `src/pipeline.py`, còn Backlog ghi `inference.py`. Chuyển cả hai vào `src/pipeline/` như trên để tránh trùng tên và mỗi người biết file của mình.

---

## 6. Cập nhật Backlog đề xuất

### 6.1 Task cần thêm

| Nhóm | Task | Phụ trách | Tuần | Ưu tiên | Deliverable | Definition of Done |
|---|---|---|---|---|---|---|
| Planning | Chốt product spec + user stories | All | 1 | P0 | `docs/product_spec.md` | Cả nhóm đọc và đồng ý use case "che mặt" |
| Integration | Viết `schemas.py` và `pipeline.yaml` | Nguyên | 2 | P0 | Schema + config | Mọi module import chung được kiểu `Face`/`Detection` |
| Integration | Module crop + mask mapping + unit test | Nguyên | 2 | P0 | `crop.py`, `mask_mapping.py` | Test đạt với ≥ 3 mặt, mặt sát biên |
| Integration | Quy tắc fallback ellipse | Nguyên | 3 | P0 | Logic trong `inference.py` | Mặt nhỏ/mask lỗi vẫn bị che |
| Integration | Báo cáo JSON + tab Phát hiện/Phân vùng | Nguyên | 3 | P0 | `visualize.py` | JSON đúng schema mục 3.5 |
| Integration | Tab So sánh Box vs Mask | Nguyên | 3 | P1 | UI tab So sánh | Hiện 2 ảnh + % diện tích bị che |
| Planning | Wireframe UI (mục 4) | Nguyên + 1 người review | 1 | P0 | `docs/ui_wireframes.md` | Cả nhóm duyệt trước khi code UI |
| Data | Chốt định nghĩa lớp "mặt" từ CelebAMask-HQ | Huy + Hương | 1 | P0 | Data card | Một định nghĩa dùng cho train/đánh giá/demo |
| Data | Bộ ảnh mẫu có thể phân phối lại | Huy | 2 | P1 | `samples/` | Ghi rõ nguồn/giấy phép từng ảnh |
| Evaluation | Recall theo kích thước mặt (nhỏ/vừa/lớn) | Yến + Uyên | 2 | P1 | Bảng Recall theo size | Có số liệu hỗ trợ chọn `conf_threshold` |
| Testing | Chỉnh ngưỡng fallback (`min_face_px`, `min_mask_ratio`) | Việt | 3 | P1 | Ghi chú trong Optimization report | Có bằng chứng trên tập val |

### 6.2 Task cần sửa

| Task hiện tại | Sửa |
|---|---|
| "Chốt yêu cầu, phạm vi và ứng dụng che mặt" | Gắn với `docs/product_spec.md`, ghi rõ đã bỏ "nhận diện danh tính" |
| "Định nghĩa interface giữa các module" (chưa có tuần) | Gán **Tuần 2**, deliverable = mục 3.3 + 3.6 |
| Các task Integration/Testing/Reporting để trống Tuần | Gán tuần cụ thể |
| Task "Viết word cho project" chưa có đủ cột | Bổ sung ưu tiên, deliverable, DoD |
| Sheet Document, dòng **Scope** ("rồi trả về bounding box") | Sửa thành: trả về **ảnh đã che + mask + JSON**, box chỉ là bước trung gian |

### 6.3 Cân bằng khối lượng

Dashboard hiện cho thấy Hương có 8 task (cộng thêm báo cáo Word), Nguyên 6, Việt 5, trong khi các thành viên khác ít hơn. Nên cân nhắc chuyển một phần (ví dụ phần crop/mask mapping hoặc viết báo cáo) sang người có ít task hơn. Mục "Báo cáo cần bàn lại để phân" trong sheet cũng đang chờ quyết định này.

---

## 7. Rủi ro, giới hạn và lưu ý

| Rủi ro | Mô tả | Giảm thiểu |
|---|---|---|
| **Domain gap** | U-Net train trên CelebAMask-HQ (ảnh mặt chân dung, rõ nét), nhưng demo trên mặt nhỏ, mờ, nghiêng trong đám đông | Margin khi crop, ngưỡng `min_face_px`, fallback ellipse; nêu rõ trong báo cáo. Đo Dice trên tập crop thực tế (task "GT crop vs predicted crop") |
| **Lỗi lan truyền** | Detection sai thì mask sai theo | Tab so sánh crop GT vs crop dự đoán (task của Yến) |
| **Bỏ sót mặt** | Công cụ che mặt mà bỏ sót thì lộ danh tính | Ưu tiên Recall, ngưỡng thấp, người dùng kiểm tra và thêm/bỏ chọn |
| **Giấy phép dữ liệu** | WIDER FACE và CelebAMask-HQ có điều kiện sử dụng riêng (thường giới hạn nghiên cứu) | Hoàn thành task "Khảo sát giấy phép"; không đưa ảnh dataset vào repo; `samples/` chỉ dùng ảnh được phép |
| **Quyền riêng tư của người dùng** | Ảnh người dùng upload chứa mặt thật | Xử lý trong phiên, không lưu ra đĩa/không gửi đi đâu, ghi rõ trên UI |
| **Che mờ ≠ ẩn danh tuyệt đối** | Blur nhẹ có thể bị làm rõ lại; không đảm bảo pháp lý | Mặc định độ mạnh đủ cao, có chế độ `solid`; ghi rõ trong README: công cụ hỗ trợ, cần kiểm tra thủ công |
| **Video nhấp nháy** | Xử lý từng khung độc lập | Nêu là giới hạn; tracking nằm ngoài phạm vi |

---

## 8. Gợi ý trả lời khi cô hỏi (vấn đáp)

1. **"Sản phẩm của nhóm là gì?"**
   Công cụ che mặt cho ảnh đông người. Người dùng upload ảnh, chọn kiểu che, tải ảnh kết quả. Hệ thống tự phát hiện (RetinaFace), tách vùng mặt (U-Net) và che theo hình dạng khuôn mặt.

2. **"Sao không chỉ dùng bounding box?"**
   Box che thừa nền và che sót góc mặt nghiêng. Tab So sánh cho thấy diện tích bị che theo mask nhỏ hơn. Box vẫn cần làm bước trung gian để U-Net chỉ xử lý vùng mặt.

3. **"Module nào nhận gì, trả gì?"**
   Mở mục 3.3: mỗi bước có input, output, shape, hệ tọa độ và trường hợp biên. Pipeline có hai hàm chính: `analyze()` (phát hiện + phân vùng) và `apply_privacy()` (che theo lựa chọn).

4. **"Nếu detection bỏ sót thì sao?"**
   Đó là lỗi nặng nhất của công cụ này nên ngưỡng mặc định đặt thấp để ưu tiên Recall. Người dùng thấy tab Phát hiện và bỏ chọn mặt thừa. Nhóm có đo Recall theo kích thước mặt để chọn ngưỡng.

5. **"Mặt quá nhỏ hoặc mask lỗi thì sao?"**
   Dùng ellipse trong bbox để vẫn che, báo cáo ghi `status = fallback_ellipse` và lý do.

6. **"Mỗi người chịu trách nhiệm gì?"**
   Xem cột Phụ trách ở mục 5 (mỗi thư mục có tên người). Mỗi người giải thích được module và hợp đồng dữ liệu của mình.

---

## 9. Việc cần cả nhóm chốt trong buổi họp tới

- [ ] Use case chính là **che mặt trước khi chia sẻ ảnh** (bỏ "camera an ninh / nhận diện").
- [ ] Tên sản phẩm (FaceShield chỉ là tên tạm).
- [ ] Gradio hay Streamlit.
- [ ] Định nghĩa lớp "mặt" từ CelebAMask-HQ (có tai không).
- [ ] `crop_size` 256 hay 512 (ảnh hưởng tốc độ và chất lượng mask).
- [ ] Phân công lại khối lượng (mục 6.3) và người viết báo cáo Word.
- [ ] Gán tuần cho các task Integration/Testing/Reporting còn trống.
