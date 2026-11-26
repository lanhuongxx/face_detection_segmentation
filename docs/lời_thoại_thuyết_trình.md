# Kịch bản thuyết trình – Phát hiện và phân vùng khuôn mặt trong ảnh đông người

**Nhóm 5 · Xử lý ảnh và Thị giác máy tính**  
**Thời lượng dự kiến:** 9 phút 40 giây, gồm 1 phút 55 giây demo.

## Slide 1. Giới thiệu đề tài | 0:00–0:15

Em chào cô và các bạn. Nhóm em trình bày đề tài phát hiện và phân vùng
khuôn mặt trong ảnh đông người bằng RetinaFace và U-Net ResNet34.

## Slide 2. Bài toán và giải pháp | 0:15–0:55

Trong một ảnh nhóm, có người đứng gần camera nên thấy rõ mặt, cũng có
người đứng xa hoặc bị che một phần. Nhóm em cần trả lời hai câu hỏi.
Thứ nhất, khuôn mặt nằm ở đâu trong ảnh? RetinaFace trả về một khung
bao, hay bounding box, cho từng mặt. Thứ hai, trong khung đó, pixel nào
thật sự thuộc vùng mặt? U-Net tạo mask để xác định vùng này. Khi có mask,
ứng dụng có thể tô đúng vùng cần xem thay vì tô cả khung chữ nhật.

## Slide 3. Luồng xử lý của hệ thống | 0:55–1:45

Đầu vào là một ảnh hoặc một khung hình người dùng chọn từ video.
RetinaFace dự đoán các bbox và điểm confidence. Sau khi lọc bbox,
hệ thống cắt từng mặt và đưa crop về kích thước 256 × 256 cho U-Net.

U-Net tạo mask trên từng crop. Hệ thống đưa mask về đúng vị trí trên
ảnh hiển thị, sau đó Streamlit cho người dùng xem bbox, mask và tải
báo cáo JSON. Luồng này có một điểm quan trọng khi đọc kết quả.
Nếu RetinaFace bỏ sót một mặt thì U-Net sẽ không nhận được crop
của mặt đó để phân vùng.

## Slide 4. Dữ liệu và tiền xử lý | 1:45–2:55

Nhóm dùng hai bộ dữ liệu vì hai tác vụ cần hai loại nhãn. Với RetinaFace,
nhóm dùng WIDER FACE, có nhãn bounding box. Nhóm lấy cùng 200 ảnh
validation, gồm 2.568 bbox hợp lệ, để so sánh các cấu hình detector.

Với U-Net, nhóm dùng CelebAMask-HQ, có mask theo từng bộ phận trên mặt.
Trong 6.000 ảnh được chọn, 4.800 ảnh dùng để train, 600 ảnh dùng để chọn
checkpoint và 600 ảnh dùng để đánh giá cuối.

Ở bước tiền xử lý, nhóm gộp các nhãn như da, mắt, mũi và môi thành một
mask mặt. Ảnh và nhãn được cắt bằng cùng một khung rồi resize về 256 × 256
để các pixel còn tương ứng. Ảnh dùng nội suy bilinear. Mask dùng
nearest-neighbor để giữ nhãn mặt và nền. Nhóm cũng giảm độ phân giải
của một phần crop train để mô phỏng ảnh bị mất chi tiết.

## Slide 5. Kiến trúc mô hình | 2:55–3:55

[Chỉ hàng RetinaFace, sau đó lần theo đường nối xuống crop của U-Net.]

Ở hàng trên là RetinaFace tiền huấn luyện. Mô hình trả bbox và điểm
confidence, sau đó các bbox được lọc trước khi đưa sang bước crop.
Nhóm đã so sánh hai backbone là MobileNet 0.25 và ResNet50. Kết quả
so sánh sẽ được trình bày ở slide 7.

Ở hàng dưới, encoder ResNet34 trích xuất các bản đồ đặc trưng từ crop.
Decoder U-Net dùng các đặc trưng đó để khôi phục độ phân giải và tạo
mask mặt. Đường nét đứt trên sơ đồ là skip connections. Nó chuyển thêm
đặc trưng từ encoder sang decoder ở nhiều mức để giữ thông tin không gian,
chẳng hạn chi tiết gần đường viền mặt. Encoder khởi tạo từ ImageNet.
Nhóm train 8 epoch với AdamW và hàm mất mát BCE cộng Dice loss.

Ghi chú khi được hỏi về embedding: ô “Bản đồ đặc trưng” là feature maps
phục vụ phân vùng. ArcFace tạo vector embedding để so khớp danh tính là
nội dung Lab 5. Pipeline coursework đầu ra là bbox và mask.

## Slide 6. Cách đánh giá | 3:55–4:50

Với detection, nhóm so bbox dự đoán với bbox chuẩn. Một dự đoán được tính
đúng khi đạt IoU ít nhất 0,50 và bbox chuẩn đó chưa được ghép. Precision
cho biết bao nhiêu dự đoán là đúng. Recall cho biết nhóm tìm được bao
nhiêu trong số mặt có nhãn. F1 tổng hợp hai chỉ số này.

Với segmentation, nhóm so mask dự đoán với mask chuẩn trên 600 crop
test. Mask được nhị phân hóa ở ngưỡng 0,50 rồi chấm bằng Dice và IoU.
Cả hai đo độ chồng lấn vùng mask. Đây là kết quả trên crop, còn độ bao phủ
trên ảnh đông người phụ thuộc thêm vào bước phát hiện.

## Slide 7. Kết quả RetinaFace | 4:50–6:20

[Chỉ vào biểu đồ so sánh backbone trước, rồi chuyển sang nhóm dưới 20 px
trên biểu đồ Recall theo kích thước.]

Nhóm chạy hai backbone trên đúng cùng 200 ảnh, giữ confidence 0,40
và giới hạn 200 dự đoán mỗi ảnh. Khi chuyển từ MobileNet sang ResNet50,
F1 tăng từ 61,96% lên 69,27%. Recall tăng từ 49,22% lên 56,27%.
Precision của ResNet50 đạt 90,09%.

Khi chia theo kích thước mặt, nhóm thấy cải thiện lớn tập trung ở mặt
rất nhỏ. ResNet50 tìm thêm 181 mặt đúng. Trong đó, 131 mặt thuộc nhóm
có cạnh ngắn bbox dưới 20 pixel sau resize. Dù vậy, Recall của nhóm này
vẫn chỉ khoảng 36%, nên đây là trường hợp hệ thống còn bỏ sót nhiều.

Thời gian phát hiện và lọc bbox tăng từ 62,77 lên 73,18 mili giây mỗi ảnh,
tức khoảng 10 mili giây. Số này đo riêng bước detection. Với mức cải thiện
về chất lượng, nhóm chọn ResNet50 cho bản demo.

## Slide 8. Kết quả U-Net | 6:20–7:10

Sau hậu xử lý, U-Net đạt Dice 0,9794 và IoU 0,9604 trên 600 crop test
CelebAMask-HQ. So với mask thô, hai điểm tăng nhẹ. Hậu xử lý giúp loại
vùng nhỏ rời rạc và lấp lỗ trong mask.

Kết quả cho thấy vùng dự đoán chồng lấn tốt với nhãn trên crop test.
Khi đưa vào ảnh đông người, điều kiện khó hơn vì mặt có thể nhỏ hoặc
bị che, và bước detection có thể bỏ sót. Sau đây nhóm minh họa cách
hai mô hình phối hợp trên một ảnh nhóm.

## Slide 9. Demo Streamlit | 7:10–9:05

[Trước giờ trình bày, mở app và chuẩn bị sẵn ảnh nhóm. Giữ slide có hai
ảnh bbox và mask làm phương án dự phòng.]

1. **Tải ảnh và bấm Phân tích.**  
   “Đây là ảnh nhóm có bảy khuôn mặt. Em tải ảnh vào và chạy phân tích
   với cấu hình RetinaFace ResNet50 đã chọn.”

2. **Mở tab Phát hiện.**  
   “Ở đây, mỗi khung là một khuôn mặt được RetinaFace phát hiện.
   Điểm hiển thị trên khung là confidence của dự đoán.”

3. **Chuyển sang tab Phân vùng.**  
   “Khi chuyển sang phân vùng, vùng màu cho biết các pixel được xác định
   thuộc vùng mặt. Mỗi mặt có mask riêng được đưa về vị trí tương ứng
   trong ảnh.”

4. **Chỉ nút tải JSON.**  
   “Người dùng có thể tải báo cáo JSON để xem số mặt phát hiện,
   tọa độ bbox và trạng thái phân vùng.”

Nếu JSON của lượt demo mới xác nhận bảy mặt đều có trạng thái ok,
chốt: “Ở ca này, hệ thống phát hiện 7 trên 7 mặt và tạo mask U-Net
cho cả bảy.”

Nếu app chậm, quay lại slide và chỉ trực tiếp ảnh bbox bên trái,
ảnh mask bên phải. Tiếp tục sang kết luận khi hết thời gian demo.

## Slide 10. Kết luận | 9:05–9:40

Nhóm em đã tích hợp RetinaFace và U-Net để phát hiện mặt, tạo mask
và xuất báo cáo JSON trên Streamlit. Qua thực nghiệm, ResNet50 giúp
tìm được nhiều mặt hơn, đặc biệt ở nhóm dưới 20 pixel. Tuy vậy, mặt
rất nhỏ vẫn là trường hợp dễ bị bỏ sót. Hướng tiếp theo của nhóm là
đánh giá mask trên ảnh đông người có nhãn và kiểm tra thêm khả năng
phát hiện mặt nhỏ. Em cảm ơn cô và các bạn đã lắng nghe.