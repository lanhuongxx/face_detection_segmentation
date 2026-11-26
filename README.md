# Face Detection & Segmentation

Phần tích hợp end-to-end của hệ thống phát hiện, phân vùng và che mờ khuôn mặt theo mask:
**RetinaFace → Crop → U-Net → Mask Map-back → Privacy Filter / Video / Streamlit Demo**.

## Cấu trúc thư mục

```text
.
├── configs/
│   └── default.json            # Cấu hình mặc định cho pipeline, blur, video
├── docs/
│   └── api_contract.md         # API contract giữa Detection, Crop, U-Net, Mask Map-back
├── models/                     # Thư mục đặt trọng số RetinaFace / U-Net (.pth)
├── outputs/                    # Thư mục lưu ảnh/video kết quả
├── src/
│   ├── contracts.py            # Dataclasses & Protocols chuẩn giữa các module
│   ├── inference.py            # Ghép Detection -> Crop -> U-Net -> Mask Map-back
│   ├── pipeline.py             # Facade tái xuất FacePrivacyPipeline
│   ├── privacy_filter.py       # Chế độ hiển thị mask, blur/pixelate theo mask, lưu ảnh
│   └── video_pipeline.py       # Đọc/ghi video ngắn và giữ nguyên FPS gốc khi xuất
├── tests/
│   ├── test_contracts.py
│   ├── test_inference.py
│   ├── test_privacy_filter.py
│   └── test_video_pipeline.py
├── app.py                      # Ứng dụng Streamlit xử lý ảnh & video
└── requirements.txt
```

## Cài đặt & chạy nhanh

```bash
pip install -r requirements.txt
pytest -q
streamlit run app.py
```

## Tích hợp với module của các thành viên khác

- `src/detection/retinaface.py`: Chỉ cần cài đặt method `detect(image)` trả về `list[FaceDetection]` hoặc `list[dict]` (có `bbox`, `score`, `landmarks`) — `RetinaFaceDetectorAdapter` trong [inference.py](src/inference.py) sẽ tự động nạp.
- `src/segmentation/unet.py`: Cung cấp class `UNet` (PyTorch `nn.Module`) hoặc truyền checkpoint `.pth` vào `PipelineConfig(unet_checkpoint=...)` — `UNetSegmentorAdapter` sẽ tự động suy luận trên từng crop và map-back về đúng tọa độ ảnh gốc.
- Khi chưa có trọng số U-Net hoặc thư viện `retina-face`, hệ thống tự động dùng fallback (Haar Cascade + heuristic face mask) để có thể kiểm thử và chạy demo ngay.
