# Thiết kế hệ thống suy luận phát hiện và phân vùng khuôn mặt

## 1. Define Problem

**Tên đề tài** — Xây dựng hệ thống suy luận phát hiện và phân vùng khuôn mặt trong ảnh đông người bằng RetinaFace và U-Net.

**Bài toán** — Trên một ảnh đông người, người dùng cần biết mô hình phát hiện những khuôn mặt nào và vùng pixel nào được dự đoán là khuôn mặt. Các trường hợp khó gồm khuôn mặt nhỏ, góc nghiêng và che khuất.

**Người dùng và tình huống sử dụng** — Người kiểm tra ảnh từ camera an ninh có thể đưa một khung hình vào ứng dụng để xem và lưu kết quả. MVP nhận một tệp ảnh JPG hoặc PNG mỗi lượt. Luồng webcam hoặc video trực tiếp là hạng mục mở rộng đề xuất.

**Mục tiêu bàn giao** — Một ứng dụng Streamlit chạy được trên máy nhóm. Người dùng tải ảnh, chạy suy luận, đối chiếu ảnh gốc với ảnh kết quả và tải ảnh PNG đã đánh dấu. Nhóm bàn giao mã nguồn, hướng dẫn chạy và ảnh mẫu để người khác kiểm tra lại.

## 2. Design System

### 2.1. Đầu vào và đầu ra của hệ thống

| Cấp xử lý | Đầu vào | Đầu ra |
| --- | --- | --- |
| RetinaFace | Ảnh gốc đã đọc thành công | Danh sách bounding box và điểm tin cậy. |
| U-Net | Một ảnh vùng mặt đã được cắt và chuẩn hóa | Bản đồ xác suất vùng mặt. Hệ thống chuyển bản đồ thành mask nhị phân. |
| Pipeline suy luận | Ảnh gốc, hai mô hình đã nạp và cấu hình suy luận | Danh sách kết quả theo từng khuôn mặt, gồm box, điểm tin cậy và mask đã ánh xạ về tọa độ ảnh gốc. Pipeline cũng tạo ảnh đã vẽ box và phủ màu theo mask. |
| Ứng dụng Streamlit | Tệp ảnh và thao tác **Xử lý ảnh** | Ảnh gốc, ảnh kết quả, số khuôn mặt phát hiện được và tệp PNG để tải xuống. |

### 2.2. Xử lý một lượt từ ảnh đến kết quả

| Bước | Nhận gì | Xử lý thế nào | Trả gì |
| --- | --- | --- | --- |
| 1. Đọc ảnh | Tệp JPG hoặc PNG | Giải mã tệp và chuẩn hóa thứ tự kênh màu cho pipeline. Dừng lượt xử lý nếu tệp không đọc được. | Ảnh RGB và kích thước gốc. |
| 2. Phát hiện | Ảnh RGB | Chạy RetinaFace pretrained. Lọc theo ngưỡng phát hiện và đưa tọa độ về ảnh gốc nếu có đổi kích thước ảnh khi chạy detector. | Danh sách box và điểm tin cậy. |
| 3. Cắt vùng mặt | Ảnh RGB và box | Giới hạn box trong biên ảnh, bỏ vùng cắt rỗng. Cắt vùng mặt rồi đổi kích thước và chuẩn hóa theo cấu hình U-Net. Giữ lại tọa độ vùng cắt. | Ảnh vùng mặt đã chuẩn hóa và tọa độ vùng cắt. |
| 4. Phân vùng | Từng vùng mặt đã chuẩn hóa | Chạy U-Net với checkpoint đã chuẩn bị. Đổi bản đồ xác suất về kích thước vùng cắt, rồi áp dụng ngưỡng mask trong cấu hình. | Mask nhị phân cho từng vùng cắt. |
| 5. Ánh xạ và tổng hợp | Ảnh gốc, tọa độ vùng cắt, box, điểm tin cậy và mask | Đặt mask vào đúng vị trí của vùng cắt trên ảnh gốc. Vẽ box và phủ màu mask trên bản sao của ảnh gốc. Nếu không có box hợp lệ, bỏ qua bước phân vùng. | Danh sách dự đoán và ảnh kết quả cùng kích thước với ảnh gốc. |
| 6. Hiển thị và xuất ảnh | Danh sách dự đoán và ảnh kết quả | Đếm số khuôn mặt trong danh sách. Mã hóa ảnh kết quả thành PNG cho nút tải xuống. | Kết quả trên giao diện và tệp `ket_qua.png`. |

**Quy ước dữ liệu giữa các module** — Box dùng tọa độ pixel `[x1, y1, x2, y2]` trên ảnh gốc. Góc trên trái là gốc tọa độ, `x2` và `y2` là cận cuối không bao gồm. Mask có giá trị `1` cho pixel thuộc vùng mặt theo quy ước nhãn và `0` cho các pixel còn lại. Box và mask của cùng một khuôn mặt được giữ trong cùng một bản ghi. Số khuôn mặt hiển thị lấy từ số bản ghi hợp lệ.

**Điều kiện để chạy suy luận** — Có trọng số RetinaFace pretrained và checkpoint U-Net tương thích với bài toán phân vùng mặt. Cấu hình phải khớp checkpoint về kích thước đầu vào và chuẩn hóa ảnh. Nhóm cần ghi rõ vùng nào được tính là mặt, cách lấy checkpoint và ngưỡng đang dùng. Nếu nhóm tự huấn luyện hoặc fine-tune U-Net trên CelebAMask-HQ, công việc đó thực hiện trước khi tích hợp checkpoint vào ứng dụng.

## 3. Thiết kế giao diện cơ bản

**Màn hình duy nhất** — Tải ảnh → bấm **Xử lý ảnh** → xem hai ảnh cạnh nhau → tải kết quả.

| Vị trí | Thành phần | Nội dung và hành vi |
| --- | --- | --- |
| Phía trên | **Chọn ảnh JPG/PNG** và **Xử lý ảnh** | Nhận một tệp mỗi lượt. Hiển thị tên tệp đã chọn. Bật nút xử lý khi đã chọn tệp. |
| Cột trái | **Ảnh gốc** | Xem trước ảnh người dùng vừa chọn. |
| Cột phải | **Kết quả phát hiện và phân vùng** | Vẽ khung quanh từng khuôn mặt phát hiện được. Phủ màu bán trong suốt lên pixel thuộc mask để người xem vẫn thấy ảnh bên dưới. |
| Dưới ảnh kết quả | **Phát hiện N khuôn mặt** và **Tải ảnh kết quả (PNG)** | Hiển thị số khuôn mặt của lượt chạy hiện tại. Tệp tải xuống chứa đúng ảnh kết quả đang hiển thị. |

| Trạng thái | Phản hồi trên màn hình |
| --- | --- |
| Đang xử lý | Hiển thị “Đang phân tích ảnh…” cho đến khi có kết quả hoặc có lỗi. |
| Không phát hiện mặt | Hiển thị “Không phát hiện khuôn mặt” và ảnh kết quả không có lớp đánh dấu. |
| Tệp lỗi | Hiển thị “Không đọc được ảnh. Vui lòng chọn tệp JPG hoặc PNG hợp lệ.” |
| Thiếu hoặc lỗi checkpoint | Hiển thị thông báo không thể nạp mô hình để nhóm kiểm tra lại cấu hình. |
| Chọn ảnh mới | Xóa kết quả của lượt trước và chờ người dùng chạy ảnh mới. |

**Kịch bản demo** — Tải ảnh đông người lên và bấm xử lý. Nếu pipeline phát hiện ba khuôn mặt thì giao diện hiển thị ba khung, lớp mask tương ứng cho từng mặt và dòng “Phát hiện 3 khuôn mặt”. Bấm tải PNG rồi mở tệp để đối chiếu với ảnh trên giao diện. Chạy thêm một ảnh không có mặt để minh họa cách hệ thống xử lý kết quả rỗng.

## 4. Folder structure đề xuất

| Đường dẫn | Trách nhiệm |
| --- | --- |
| `app.py` | Điểm chạy Streamlit. Nhận tệp, gọi pipeline và hiển thị kết quả. |
| `src/detection/retinaface.py` | Nạp RetinaFace và trả danh sách box, điểm tin cậy. |
| `src/segmentation/predict.py` | Nạp U-Net, chuẩn hóa đầu vào và dự đoán bản đồ xác suất vùng mặt. |
| `src/pipeline.py` | Điều phối phát hiện, cắt mặt, phân vùng và ánh xạ mask. Trả dữ liệu dự đoán cùng ảnh kết quả. |
| `src/visualization.py` | Vẽ box và phủ màu mask lên ảnh. |
| `configs/inference.yaml` | Đường dẫn checkpoint, kích thước đầu vào, chuẩn hóa và ngưỡng suy luận. |
| `weights/` | Vị trí đặt checkpoint để chạy demo. |
| `examples/` | Ảnh mẫu được phép sử dụng và ảnh kết quả tương ứng. |
| `tests/` | Ca kiểm tra pipeline và checklist chạy demo. |
| `requirements.txt` và `README.md` | Thư viện cần cài, cách chuẩn bị checkpoint và lệnh chạy `streamlit run app.py`. |

## 5. MVP và tiêu chí nghiệm thu

**Thứ tự thực hiện** — Chuẩn bị checkpoint → chạy thử riêng RetinaFace và U-Net → tích hợp và kiểm tra tọa độ mask → nối pipeline vào Streamlit → chạy kịch bản demo.

| Ca kiểm tra | Tiêu chí đạt |
| --- | --- |
| Ảnh mẫu nhiều mặt | Ứng dụng chạy từ tải ảnh đến kết quả. Mỗi box hợp lệ đi với một mask tương ứng. Dùng ảnh mẫu để kiểm tra mask nằm đúng vùng trên ảnh gốc. |
| Ánh xạ mask | Với một vùng cắt và mask giả lập có tọa độ biết trước, pipeline đặt mask vào đúng vị trí, kể cả khi box sát biên ảnh. |
| Ảnh không có mặt và tệp lỗi | Ứng dụng báo đúng trạng thái và tiếp tục cho chọn ảnh khác. |
| Xuất PNG | Tệp mở được, cùng kích thước với ảnh gốc và chứa đúng kết quả đang hiển thị. |

**Đánh giá đi kèm** — Có thể báo cáo AP cho RetinaFace trên WIDER FACE và IoU/Dice cho U-Net trên tập kiểm thử có mask chuẩn. Với ảnh đông người, ghi nhận các ca bỏ sót mặt nhỏ, mặt bị che và các ca mask lệch. Nếu đo IoU/Dice toàn pipeline trên ảnh đông người, cần mask chuẩn cho từng khuôn mặt trong các ảnh đó.
