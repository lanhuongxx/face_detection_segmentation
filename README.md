# Face Detection & Face Segmentation

Face Detection using RetinaFace and Face Segmentation using U-Net.

## Team
6 members

## Technologies
- Python
- PyTorch
- OpenCV
- RetinaFace
- U-Net

# Cách chép và chạy pipeline

1. Nếu đã có `src/` từ trước (và có sửa, ví dụ thêm NMS): **chỉ chép `app.py` và `scripts/run_demo.py`**, đừng đè `src/`. Nếu chưa có thì giải nén cả gói vào thư mục project (`face_detection_segmentation/`).
2. Đặt model đã train ở `models/unet_best.pt` (chạy CELL 13 của notebook). Nên có cả `models/unet_best_info.json`.
3. Bật môi trường rồi chạy:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   python tests/test_pipeline_logic.py                 # 12 test logic, không cần GPU
   python scripts/run_demo.py --input samples/anh.jpg  # thử trên 1 ảnh
   streamlit run app.py                                # giao diện
   ```
4. Đổi tham số (ngưỡng, kiểu che, detector) trong `configs/pipeline.yaml`.
   Muốn bắt mặt nhỏ tốt hơn: `detection.model: retinaface_resnet50`.
