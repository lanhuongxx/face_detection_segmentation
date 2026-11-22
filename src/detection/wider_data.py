"""Đọc nhãn WIDER FACE và tạo manifest (danh sách ảnh) cho phần detection.

Mỗi ảnh trong file nhãn gồm: một dòng tên ảnh, một dòng số mặt, rồi mỗi mặt một dòng
    x y w h blur expression illumination invalid occlusion pose
Ảnh không có mặt thì vẫn có thêm 1 dòng toàn số 0 phía sau, dòng đó phải bỏ qua.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path

import numpy as np

# Chia mặt theo chiều cao bbox (pixel): nhỏ < 32 <= vừa < 96 <= lớn
DEFAULT_SIZE_GROUPS = (32, 96)


def image_key(path: str) -> str:
    """Lấy phần '<event>/<tên file>' làm khóa của ảnh.

    File nhãn ghi tên ảnh dạng '0--Parade/0_Parade_marchingband_1_5.jpg', còn predictions
    lưu đường dẫn đầy đủ trong manifest. Cắt lấy hai đoạn cuối để hai bên ghép được với nhau,
    bất kể đường dẫn gốc dùng dấu / hay \\.
    """
    parts = str(path).replace("\\", "/").split("/")
    return "/".join(parts[-2:])


def load_ground_truth(path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Đọc file nhãn WIDER, trả về {khóa ảnh: (boxes xyxy (N,4), invalid (N,) kiểu bool)}."""
    # utf-8-sig phòng trường hợp file có BOM; bỏ luôn các dòng trống
    with open(path, encoding="utf-8-sig") as f:
        lines = [line.strip() for line in f if line.strip()]
    gt = {}
    i = 0
    while i < len(lines):
        # mỗi khối gồm: tên ảnh, số mặt, rồi (số mặt) dòng nhãn
        key = image_key(lines[i])
        count = int(lines[i + 1])
        i += 2
        if count == 0:
            i += 1  # dòng giữ chỗ toàn số 0
            gt[key] = (np.zeros((0, 4)), np.zeros(0, dtype=bool))
            continue
        rows = np.array([[float(v) for v in lines[i + k].split()] for k in range(count)])
        i += count
        x, y, w, h = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
        # file nhãn lưu (x, y, w, h), đổi sang (x1, y1, x2, y2) cho tiện tính IoU
        boxes = np.stack([x, y, x + w, y + h], axis=1)
        # cột 7 là cờ invalid. Mặt invalid hoặc box rỗng thì không dùng làm đáp án
        invalid = (rows[:, 7] == 1) | (w <= 0) | (h <= 0)
        if key in gt:
            raise ValueError(f"Trùng ảnh trong file nhãn: {key}")
        gt[key] = (boxes, invalid)
    return gt


def make_manifests(root: str, tune_n: int = 500, seed: int = 42) -> None:
    """Tạo 2 manifest trong thư mục root (chứa WIDER_val, WIDER_train, wider_face_split):

    - manifest_wider_val.csv: toàn bộ WIDER val, split=test
    - manifest_wider_tune500.csv: 500 ảnh train lấy cố định (seed 42), split=tune
    """
    root = Path(root)
    ann = root / "wider_face_split"
    # sorted để danh sách id có thứ tự cố định, nếu không thì cùng seed vẫn có thể ra mẫu khác
    val_ids = sorted(load_ground_truth(ann / "wider_face_val_bbx_gt.txt"))
    train_ids = sorted(load_ground_truth(ann / "wider_face_train_bbx_gt.txt"))
    tune_ids = sorted(random.Random(seed).sample(train_ids, tune_n))

    def write(name, folder, ids, split):
        with open(root / name, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["image_path", "split"])
            for i in ids:
                p = root / folder / "images" / i
                # kiểm tra file ảnh có thật, thiếu ảnh thì dừng ngay chứ không ghi manifest hỏng
                if not p.is_file():
                    raise FileNotFoundError(f"Thiếu ảnh: {p}")
                # ghi đường dẫn tương đối so với root để manifest chuyển sang máy khác vẫn dùng được
                w.writerow([os.path.relpath(p, root).replace("\\", "/"), split])

    write("manifest_wider_val.csv", "WIDER_val", val_ids, "test")
    write(f"manifest_wider_tune{tune_n}.csv", "WIDER_train", tune_ids, "tune")
    print(f"Val: {len(val_ids)} ảnh, tune: {len(tune_ids)} ảnh. Đã ghi vào {root}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Tạo manifest WIDER val và tune.")
    p.add_argument("--root", default="data/wider_face")
    make_manifests(p.parse_args().root)
