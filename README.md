# Face Detection & Segmentation

Hệ thống phát hiện và phân vùng khuôn mặt bằng **RetinaFace** và **U-Net**. Pipeline nhận ảnh RGB, phát hiện từng khuôn mặt, cắt vùng mặt, dự đoán mask, ánh xạ mask về ảnh gốc và xuất kết quả trực quan cùng báo cáo JSON.

Giao diện Streamlit hỗ trợ ảnh tĩnh, camera và video. Chức năng che mặt đã được tích hợp trong pipeline nhưng đang tắt trong giao diện hiện tại; giao diện chủ yếu hiển thị phát hiện, phân vùng và tải ảnh mask/báo cáo.

## Tính năng

- Phát hiện nhiều khuôn mặt bằng RetinaFace.
- Phân vùng khuôn mặt bằng U-Net.
- Hỗ trợ RetinaFace mobile hoặc ResNet50.
- NMS, giới hạn số mặt và lọc kích thước ảnh.
- Mask dự phòng dạng ellipse nếu mặt quá nhỏ hoặc mask U-Net không hợp lệ.
- Chạy trên CPU hoặc GPU CUDA.
- Xuất overlay phát hiện, overlay mask và báo cáo JSON.
- Kiểm thử logic pipeline bằng detector và segmenter giả lập.

## Công nghệ

- Python
- PyTorch, torchvision
- OpenCV
- Pillow
- facexlib
- Streamlit
- NumPy
- PyYAML

## Cấu trúc project

```text
face_detection_segmentation/
├── app.py                         # Giao diện Streamlit
├── configs/
│   └── pipeline.yaml              # Cấu hình inference
├── src/
│   ├── detection/                 # RetinaFace
│   ├── segmentation/              # U-Net
│   ├── pipeline/                  # Crop, mask, inference, export
│   ├── utils/                     # Cấu hình và xử lý I/O
│   ├── run_demo.py                # Chạy pipeline trên một ảnh
│   └── evaluate_wider_face.py     # Đánh giá trên WIDER FACE
├── tests/
│   └── test_pipeline_logic.py     # Test không cần GPU
├── celebamask_unet_v3.ipynb       # Huấn luyện U-Net
├── wider_face_evaluation.ipynb    # Notebook đánh giá
├── docs/                          # Thiết kế sản phẩm và hệ thống
├── data/                          # Dữ liệu và kết quả huấn luyện
├── models/                        # Checkpoint trọng số
├── outputs/                       # Kết quả sinh ra
└── requirements.txt
```

Các thư mục `data/`, `models/` và `outputs/` có thể chứa dữ liệu lớn hoặc tệp sinh ra nên không được lấy toàn bộ bằng lệnh liệt kê cây thông thường.

## Yêu cầu

- Python 3.10 trở lên.
- Trên Windows, khuyến nghị sử dụng `.venv`.
- GPU NVIDIA là tùy chọn; pipeline tự chuyển sang CPU nếu CUDA không khả dụng.
- Cần checkpoint `models/unet_best.pt`. Chạy cell lưu model trong notebook huấn luyện hoặc đặt checkpoint tương thích vào đường dẫn này.

## Cài đặt

```powershell
cd C:\Users\NAO\face_detection_segmentation

py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nếu lỗi `facexlib` hoặc tải RetinaFace, hãy chắc chắn đã cài đầy đủ PyTorch tương thích hệ thống trước rồi mới cài các thư viện còn lại.

## Chạy ứng dụng

```powershell
streamlit run app.py
```

Mở địa chỉ Streamlit được in ra terminal, sau đó tải ảnh JPG/PNG/WebP hoặc chọn ảnh mẫu.

## Chạy pipeline bằng dòng lệnh

```powershell
python src/run_demo.py --input duong/dan/anh.jpg
```

Các tùy chọn:

```powershell
python src/run_demo.py --input duong/dan/anh.jpg --conf 0.5 --mode blur --strength 6 --region mask --out outputs/results
```

Kết quả được lưu thành PNG box, PNG mask, ảnh đã xử lý và báo cáo JSON trong thư mục đầu ra.

## Kiểm thử

Các test logic dùng dữ liệu giả và không cần tải model hoặc GPU:

```powershell
python tests/test_pipeline_logic.py
```

Hoặc chạy bằng pytest nếu đã cài pytest:

```powershell
python -m pytest tests -q
```

## Đánh giá RetinaFace trên WIDER FACE

Tải dữ liệu WIDER FACE validation và đặt theo cấu trúc:

```text
data/raw/WIDER_FACE/
├── wider_face_split/
│   └── wider_face_val_bbx_gt.txt
└── WIDER_val/
    └── images/
```

Sau đó chạy:

```powershell
python src/evaluate_wider_face.py --limit 200
```

Lệnh tính Precision, Recall, TP, FP và FN trên một mẫu ảnh được cố định bằng seed.

## Cấu hình

Các tham số inference nằm trong [`configs/pipeline.yaml`](configs/pipeline.yaml):

- `device`: `auto`, `cuda` hoặc `cpu`.
- `detection.model`: `retinaface_mobile0.25` hoặc `retinaface_resnet50`.
- `detection.conf_threshold`: ngưỡng phát hiện.
- `crop.min_face_px`: ngưỡng kích thước mặt được phân vùng.
- `segmentation.weights`: đường dẫn checkpoint U-Net.
- `segmentation.mask_threshold`: ngưỡng tạo mask nhị phân.
- `privacy.mode`: `blur`, `pixelate` hoặc `solid`.
- `privacy.region`: `mask`, `bbox` hoặc `ellipse`.

Để bắt mặt nhỏ tốt hơn, có thể đổi:

```yaml
detection:
  model: retinaface_resnet50
```

Lưu ý: model nặng hơn và có thể chậm hơn trên CPU.

## Giới hạn

- Hệ thống phát hiện và phân vùng khuôn mặt; chưa nhận diện danh tính.
- Chất lượng mask phụ thuộc checkpoint U-Net và các mặt nhỏ/bị che khuất.
- Kết quả che mặt nên được kiểm tra thủ công trước khi chia sẻ.
- Đây là công cụ hỗ trợ, không thay thế đánh giá bảo mật hoặc tuân thủ pháp lý.

## Tài liệu

- [Thiết kế sản phẩm và hệ thống](docs/PRODUCT_AND_SYSTEM_DESIGN.md)
- [Thiết kế hệ thống suy luận](docs/thiet_ke_face_detection_segmentation.md)
