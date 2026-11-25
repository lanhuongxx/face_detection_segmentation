"""Chạy pipeline trên một ảnh từ dòng lệnh, không cần Streamlit. Dùng để kiểm tra nhanh trên GPU.
Ví dụ:  python scripts/run_demo.py --input samples/anh.jpg --mode blur --strength 6"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from src.pipeline.inference import FacePipeline
from src.pipeline.preprocess_input import load_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", default="outputs/results")
    parser.add_argument("--conf", type=float, default=None, help="ngưỡng phát hiện (mặc định lấy từ config)")
    parser.add_argument("--mode", default="blur", choices=["blur", "pixelate", "solid"])
    parser.add_argument("--strength", type=int, default=6)
    parser.add_argument("--region", default="mask", choices=["mask", "bbox", "ellipse"])
    args = parser.parse_args()

    image, meta = load_image(args.input)
    pipeline = FacePipeline()
    result = pipeline.analyze(image, conf_threshold=args.conf, source={**meta, "type": "image"})
    final = pipeline.apply_privacy(result, [f.id for f in result.faces], args.mode, args.strength, args.region)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.input).stem
    Image.fromarray(result.overlay_boxes).save(out / f"{stem}_boxes.png")
    Image.fromarray(result.overlay_masks).save(out / f"{stem}_masks.png")
    Image.fromarray(final).save(out / f"{stem}_result.png")
    (out / f"{stem}_report.json").write_text(
        json.dumps(pipeline.report(result), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(result.faces)} khuôn mặt · {result.timing_ms['total']:.0f} ms · đã lưu vào {out.resolve()}")


if __name__ == "__main__":
    main()
