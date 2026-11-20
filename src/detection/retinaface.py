"""Inference RetinaFace pretrained và chuẩn hóa kết quả về hợp đồng chung."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np


def bbox_to_int(
    bbox: Sequence[float], image_shape: Sequence[int]
) -> tuple[int, int, int, int] | None:
    """Đổi bbox float sang chỉ số crop NumPy, trả None nếu vùng crop rỗng."""
    height, width = image_shape[:2]
    x1, y1, x2, y2 = bbox

    left = min(max(math.floor(x1), 0), width)
    top = min(max(math.floor(y1), 0), height)
    right = min(max(math.ceil(x2), 0), width)
    bottom = min(max(math.ceil(y2), 0), height)

    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


class RetinaFaceDetector:
    """Phát hiện khuôn mặt bằng RetinaFace pretrained của FaceXLib."""

    # Giới hạn này tránh cấp phát tensor quá lớn cho ảnh đầu vào độ phân giải cao.
    _DEFAULT_MAX_SIZE = 2150
    # Một số convolution của backbone cần ảnh có cạnh tối thiểu lớn hơn ảnh thumbnail.
    _MIN_INPUT_SIDE = 32
    _SUPPORTED_MODELS = {"resnet50", "mobile0.25"}

    def __init__(
        self,
        model_name: str = "resnet50",
        device: str = "auto",
        conf_thres: float = 0.5,
        nms_thres: float = 0.4,
        max_size: int | None = None,
    ) -> None:
        """Tải model pretrained và kiểm tra cấu hình inference."""
        # Kiểm tra cấu hình sớm để lỗi tham số không bị nhầm với lỗi tải model.
        if model_name not in self._SUPPORTED_MODELS:
            raise ValueError(
                f"model_name phải là một trong {sorted(self._SUPPORTED_MODELS)}."
            )
        if not 0.0 <= conf_thres <= 1.0:
            raise ValueError("conf_thres phải nằm trong [0, 1].")
        if not 0.0 <= nms_thres <= 1.0:
            raise ValueError("nms_thres phải nằm trong [0, 1].")
        if max_size is not None and max_size < self._MIN_INPUT_SIDE:
            raise ValueError(f"max_size phải >= {self._MIN_INPUT_SIDE} hoặc là None.")
        if device != "auto" and not isinstance(device, str):
            raise TypeError("device phải là chuỗi 'auto', 'cpu' hoặc tên thiết bị PyTorch.")

        # Import tại đây để chỉ cần cài PyTorch/FaceXLib khi thực sự khởi tạo detector.
        import torch
        from facexlib.detection import init_detection_model

        # Chọn CUDA nếu có; nếu người dùng yêu cầu CUDA nhưng máy không hỗ trợ thì báo rõ.
        if device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            resolved_device = device
            if resolved_device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError(
                    f"Yêu cầu device='{resolved_device}' nhưng CUDA không khả dụng."
                )

        self.model_name = model_name
        self.device = resolved_device
        self.conf_thres = float(conf_thres)
        self.nms_thres = float(nms_thres)
        self.max_size = max_size
        self._torch = torch
        # FaceXLib tự tải weights pretrained lần đầu; không thực hiện huấn luyện ở đây.
        self._model = init_detection_model(
            model_name=f"retinaface_{model_name}",
            device=resolved_device,
        )

    @staticmethod
    def _to_bgr(image: np.ndarray) -> np.ndarray:
        """Chuẩn hóa ảnh xám/BGR/BGRA thành ảnh BGR uint8."""
        if not isinstance(image, np.ndarray):
            raise ValueError("Ảnh đầu vào phải là numpy.ndarray, không được là None.")
        if image.size == 0:
            raise ValueError("Ảnh không được rỗng.")
        if image.dtype.kind not in "uif":
            raise ValueError(
                "Ảnh có kiểu dữ liệu không hợp lệ; chỉ chấp nhận số nguyên hoặc số thực."
            )
        if not np.isfinite(image).all():
            raise ValueError("Ảnh chứa NaN hoặc infinity.")

        # FaceXLib/OpenCV cần pixel 8-bit; ảnh float [0, 1] được đổi sang [0, 255].
        if image.dtype != np.uint8:
            values = image.astype(np.float32, copy=False)
            if image.dtype == np.uint16:
                values = values / 257.0
            elif values.size and values.min() >= 0.0 and values.max() <= 1.0:
                values = values * 255.0
            image = np.clip(values, 0.0, 255.0).astype(np.uint8)

        # Giữ nguyên thứ tự BGR theo OpenCV; BGRA chỉ bỏ alpha, không đổi kênh màu.
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 1:
            image = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, :3]
        elif image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Ảnh phải có shape HxW, HxWx1, HxWx3 hoặc HxWx4.")

        if image.shape[0] == 0 or image.shape[1] == 0:
            raise ValueError("Ảnh không được rỗng.")
        return np.ascontiguousarray(image)

    def _prepare_image(self, image: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Giới hạn kích thước ảnh và trả lại hệ số scale từng trục."""
        image = self._to_bgr(image)
        height, width = image.shape[:2]
        # Ảnh lớn được thu nhỏ; ảnh quá nhỏ được phóng lên để backbone xử lý ổn định.
        max_size = self.max_size or self._DEFAULT_MAX_SIZE
        largest_side = max(height, width)

        if largest_side > max_size:
            scale = max_size / largest_side
        elif min(height, width) < self._MIN_INPUT_SIDE:
            scale = min(
                self._MIN_INPUT_SIDE / min(height, width),
                max_size / largest_side,
            )
        else:
            scale = 1.0

        # Dùng kích thước đã làm tròn để tính scale thực tế, tránh lệch tọa độ vài pixel.
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        if (resized_width, resized_height) != (width, height):
            image = cv2.resize(
                image,
                (resized_width, resized_height),
                interpolation=cv2.INTER_LINEAR,
            )
        # Làm tròn có thể khiến scale ngang và dọc hơi khác nhau, nên lưu riêng từng trục.
        scale_x = resized_width / width
        scale_y = resized_height / height
        return image, scale_x, scale_y

    def detect(self, image_bgr: np.ndarray) -> list[dict[str, Any]]:
        """Trả các mặt theo bbox, score và 5 landmarks ở tọa độ ảnh gốc."""
        image, scale_x, scale_y = self._prepare_image(image_bgr)
        # Inference mode tắt gradient; ngoài tiết kiệm bộ nhớ còn tương thích với
        # FaceXLib đang chuyển tensor kết quả trực tiếp sang NumPy.
        with self._torch.inference_mode():
            predictions = self._model.detect_faces(
                image,
                conf_threshold=self.conf_thres,
                nms_threshold=self.nms_thres,
                use_origin_size=True,
            )
        # FaceXLib trả mỗi mặt thành 15 số: bbox(4), score(1), landmarks(10).
        detections = np.asarray(predictions)
        if detections.size == 0:
            return []
        if detections.ndim != 2 or detections.shape[1] != 15:
            raise RuntimeError(
                "FaceXLib trả kết quả sai định dạng: kỳ vọng N x 15 "
                "(bbox, score, 5 landmarks), nhận "
                f"{detections.shape}."
            )
        if not np.isfinite(detections).all():
            raise RuntimeError("FaceXLib trả bbox/score/landmarks không hữu hạn.")

        height, width = image_bgr.shape[:2]
        results: list[dict[str, Any]] = []
        for detection in detections:
            # Model dự đoán trên ảnh đã resize; đổi bbox ngược về pixel ảnh gốc.
            x1, y1, x2, y2 = detection[:4].astype(np.float64)
            x1 = float(np.clip(x1 / scale_x, 0, width))
            x2 = float(np.clip(x2 / scale_x, 0, width))
            y1 = float(np.clip(y1 / scale_y, 0, height))
            y2 = float(np.clip(y2 / scale_y, 0, height))
            if bbox_to_int((x1, y1, x2, y2), image_bgr.shape) is None:
                continue

            # landmarks được xếp phẳng theo [x1,y1,...,x5,y5], cần khôi phục thành 5 điểm.
            points = detection[5:15].astype(np.float64).reshape(5, 2)
            points[:, 0] = np.clip(points[:, 0] / scale_x, 0, width - 1)
            points[:, 1] = np.clip(points[:, 1] / scale_y, 0, height - 1)
            results.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "score": float(np.clip(detection[4], 0.0, 1.0)),
                    "landmarks": points.tolist(),
                }
            )
        results.sort(key=lambda face: face["score"], reverse=True)
        return results

    def detect_batch(
        self, images: list[np.ndarray]
    ) -> list[list[dict[str, Any]]]:
        """Phát hiện từng ảnh lần lượt; đầu ra giữ cùng thứ tự với danh sách đầu vào."""
        # FaceXLib không cung cấp batch API thuận tiện cho ảnh có kích thước khác nhau.
        return [self.detect(image) for image in images]

    @staticmethod
    def draw(
        image: np.ndarray, faces: list[dict[str, Any]]
    ) -> np.ndarray:
        """Vẽ bbox, confidence và landmarks lên bản sao ảnh BGR."""
        # Vẽ trên bản sao để không làm thay đổi ảnh gốc đang được pipeline sử dụng.
        canvas = RetinaFaceDetector._to_bgr(image).copy()
        for face in faces:
            x1, y1, x2, y2 = (int(round(value)) for value in face["bbox"])
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                canvas,
                f"{face['score']:.2f}",
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
            # FaceXLib quy ước landmarks gồm hai mắt, mũi và hai khóe miệng.
            for point in face.get("landmarks") or []:
                x, y = (int(round(value)) for value in point)
                cv2.circle(canvas, (x, y), 2, (0, 0, 255), -1)
        return canvas


def main(argv: list[str] | None = None) -> None:
    """Chạy demo đọc ảnh, detect và lưu ảnh debug."""
    # Giúp argparse và thông báo lỗi tiếng Việt in được trên terminal Windows cũ.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="backslashreplace")

    # Cho phép demo chọn ảnh, nơi lưu kết quả và thiết bị mà không sửa code.
    parser = argparse.ArgumentParser(description="Demo RetinaFace inference.")
    parser.add_argument("--image", required=True, help="Đường dẫn ảnh đầu vào.")
    parser.add_argument("--out", required=True, help="Đường dẫn ảnh debug đầu ra.")
    parser.add_argument("--device", default="auto", help="auto, cpu hoặc cuda.")
    parser.add_argument(
        "--conf-thres", type=float, default=0.5, help="Confidence threshold."
    )
    args = parser.parse_args(argv)

    image_path = Path(args.image)
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        # Phân biệt file không tồn tại với file có tồn tại nhưng OpenCV không giải mã được.
        if not image_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy ảnh đầu vào: {image_path}")
        raise ValueError(f"Không giải mã được ảnh đầu vào: {image_path}")

    detector = RetinaFaceDetector(
        device=args.device,
        conf_thres=args.conf_thres,
    )
    faces = detector.detect(image)
    rendered = detector.draw(image, faces)

    output_path = Path(args.out)
    # Tạo sẵn thư mục cha để người dùng có thể truyền đường dẫn output mới.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), rendered):
        raise OSError(f"Không ghi được ảnh kết quả: {output_path}")
    print(f"Đã phát hiện {len(faces)} mặt; ảnh kết quả: {output_path}")


if __name__ == "__main__":
    main()
