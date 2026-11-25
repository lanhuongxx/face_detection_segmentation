"""Streamlit application for Face Detection, Segmentation, and Privacy Filtering."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from src.contracts import PrivacyMode
from src.inference import FacePrivacyPipeline, PipelineConfig
from src.privacy_filter import PrivacyFilter, PrivacyFilterConfig
from src.video_pipeline import VideoConfig, VideoPrivacyProcessor

MODE_LABELS: dict[str, PrivacyMode] = {
    "Làm mờ mặt theo mask (Blur)": "blur",
    "Làm mờ + Khung khuôn mặt (Blur + BBox)": "blur_with_bbox",
    "Che ô vuông theo mask (Pixelate)": "pixelate",
    "Hiển thị Mask phủ màu (Mask Overlay)": "mask",
    "Chỉ hiển thị Mask (Mask Only)": "mask_only",
    "Ảnh gốc (Original)": "original",
}


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    cleaned = hex_color.lstrip("#")
    if len(cleaned) != 6:
        return (0, 200, 255)
    return (int(cleaned[0:2], 16), int(cleaned[2:4], 16), int(cleaned[4:6], 16))


@st.cache_resource(show_spinner=False)
def _load_pipeline(
    detector_backend: str,
    segmentor_backend: str,
    conf_threshold: float,
    mask_threshold: float,
    crop_margin_ratio: float,
    unet_checkpoint: str | None,
) -> FacePrivacyPipeline:
    cfg = PipelineConfig(
        detector_backend=detector_backend,
        segmentor_backend=segmentor_backend,
        conf_threshold=conf_threshold,
        mask_threshold=mask_threshold,
        crop_margin_ratio=crop_margin_ratio,
        unet_checkpoint=unet_checkpoint or None,
    )
    return FacePrivacyPipeline(config=cfg)


def _render_sidebar() -> tuple[FacePrivacyPipeline, PrivacyFilterConfig]:
    st.sidebar.header("Cấu hình Pipeline")

    detector_backend = st.sidebar.selectbox(
        "Detector Backend",
        options=["auto", "retinaface", "haar"],
        index=0,
        help="`auto` tự dùng RetinaFace nếu có, hoặc fallback sang Haar Cascade.",
    )
    conf_threshold = st.sidebar.slider("Ngưỡng tin cậy Detection", 0.10, 0.95, 0.50, 0.05)

    segmentor_backend = st.sidebar.selectbox(
        "Segmentor Backend",
        options=["auto", "heuristic", "unet"],
        index=0,
        help="`auto` dùng U-Net nếu có checkpoint, ngược lại dùng fallback heuristic.",
    )
    checkpoint_str = st.sidebar.text_input(
        "Đường dẫn U-Net checkpoint (.pth, tùy chọn)",
        value="",
        placeholder="models/unet_best.pth",
    ).strip()

    mask_threshold = st.sidebar.slider("Ngưỡng nhị phân Mask", 0.10, 0.90, 0.50, 0.05)
    crop_margin = st.sidebar.slider("Tỉ lệ mở rộng vùng Crop", 0.00, 0.40, 0.15, 0.05)

    st.sidebar.divider()
    st.sidebar.header("Chế độ hiển thị & Làm mờ")
    mode_label = st.sidebar.selectbox("Chế độ xử lý", options=list(MODE_LABELS.keys()), index=0)
    mode = MODE_LABELS[mode_label]

    blur_strength = st.sidebar.slider("Độ mạnh Gaussian Blur", 7, 99, 35, 2)
    feather_radius = st.sidebar.slider("Làm mượt biên Mask (Feather)", 1, 31, 9, 2)
    pixel_block_size = st.sidebar.slider("Kích thước ô Pixelate", 4, 48, 16, 2)
    mask_alpha = st.sidebar.slider("Độ đậm Mask Overlay", 0.10, 0.90, 0.45, 0.05)
    mask_hex = st.sidebar.color_picker("Màu phủ Mask", "#00C8FF")
    draw_bbox = st.sidebar.checkbox("Vẽ Bounding Box", value=(mode in ("mask", "blur_with_bbox")))
    draw_labels = st.sidebar.checkbox("Hiện nhãn độ tin cậy", value=True)

    try:
        pipeline = _load_pipeline(
            detector_backend=detector_backend,
            segmentor_backend=segmentor_backend,
            conf_threshold=conf_threshold,
            mask_threshold=mask_threshold,
            crop_margin_ratio=crop_margin,
            unet_checkpoint=checkpoint_str or None,
        )
    except Exception as exc:
        st.sidebar.error(f"Lỗi khởi tạo model: {exc}")
        st.stop()

    st.sidebar.caption(
        f"Đang dùng: Detector=`{pipeline.detector.backend_used}` | "
        f"Segmentor=`{pipeline.segmentor.backend_used}`"
    )

    filter_cfg = PrivacyFilterConfig(
        mode=mode,
        blur_strength=blur_strength,
        feather_radius=feather_radius,
        pixel_block_size=pixel_block_size,
        mask_color=_hex_to_rgb(mask_hex),
        mask_alpha=mask_alpha,
        draw_bbox=draw_bbox,
        draw_labels=draw_labels,
    )
    return pipeline, filter_cfg


def _render_image_tab(pipeline: FacePrivacyPipeline, filter_cfg: PrivacyFilterConfig) -> None:
    uploaded = st.file_uploader(
        "Tải lên ảnh cần xử lý",
        type=["png", "jpg", "jpeg", "webp"],
        key="image_uploader",
    )
    if uploaded is None:
        st.info("Hãy upload một ảnh (JPG, PNG, WEBP) để phát hiện, phân vùng và làm mờ khuôn mặt.")
        return

    pil_img = Image.open(uploaded).convert("RGB")
    rgb = np.asarray(pil_img, dtype=np.uint8)

    with st.spinner("Đang chạy RetinaFace -> Crop -> U-Net -> Map-back..."):
        output = pipeline.run(rgb)
        privacy_filter = PrivacyFilter(filter_cfg)
        rendered = privacy_filter.apply(rgb, output, config=filter_cfg)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Số khuôn mặt", output.num_faces)
    m2.metric("Tổng thời gian", f"{output.timings_ms.get('total_ms', 0.0):.1f} ms")
    m3.metric("Detection", f"{output.timings_ms.get('detect_ms', 0.0):.1f} ms")
    m4.metric("Segmentation + Map-back", f"{output.timings_ms.get('segment_ms', 0.0) + output.timings_ms.get('map_back_ms', 0.0):.1f} ms")

    if output.num_faces == 0:
        st.warning("Không phát hiện khuôn mặt nào trong ảnh.")

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Ảnh gốc")
        st.image(rgb, use_container_width=True)
    with col_right:
        st.subheader("Ảnh kết quả")
        st.image(rendered, use_container_width=True)

    stem = Path(uploaded.name).stem
    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            label="Tải ảnh kết quả (.png)",
            data=privacy_filter.encode_image(rendered, ext=".png"),
            file_name=f"{stem}_{filter_cfg.mode}.png",
            mime="image/png",
            use_container_width=True,
        )
    with dl2:
        st.download_button(
            label="Tải ảnh kết quả (.jpg)",
            data=privacy_filter.encode_image(rendered, ext=".jpg", quality=95),
            file_name=f"{stem}_{filter_cfg.mode}.jpg",
            mime="image/jpeg",
            use_container_width=True,
        )

    if output.faces:
        with st.expander(f"Chi tiết tọa độ từng khuôn mặt ({output.num_faces} mặt)"):
            rows = []
            for idx, face in enumerate(output.faces, start=1):
                rows.append(
                    {
                        "Face": idx,
                        "Score": round(face.detection.score, 3),
                        "BBox (x1, y1, x2, y2)": face.detection.bbox,
                        "Crop Box": face.crop_metadata.crop_box,
                        "Mask Pixels": int(face.full_mask.sum()),
                    }
                )
            st.dataframe(rows, use_container_width=True)


def _render_video_tab(pipeline: FacePrivacyPipeline, filter_cfg: PrivacyFilterConfig) -> None:
    uploaded_video = st.file_uploader(
        "Tải lên video ngắn (MP4, AVI, MOV, MKV)",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_uploader",
    )
    if uploaded_video is None:
        st.info("Upload một video ngắn để xử lý theo từng frame và giữ nguyên FPS gốc khi xuất.")
        return

    c1, c2 = st.columns(2)
    with c1:
        frame_stride = st.number_input(
            "Frame stride (1 = suy luận mọi frame)",
            min_value=1,
            max_value=10,
            value=1,
            step=1,
        )
    with c2:
        max_frames_val = st.number_input(
            "Giới hạn số frame (0 = toàn bộ video)",
            min_value=0,
            max_value=5000,
            value=0,
            step=30,
        )

    if st.button("Bắt đầu xử lý video", type="primary"):
        suffix = Path(uploaded_video.name).suffix or ".mp4"
        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = Path(tmpdir) / f"input{suffix}"
            out_path = Path(tmpdir) / f"{Path(uploaded_video.name).stem}_{filter_cfg.mode}.mp4"
            in_path.write_bytes(uploaded_video.getvalue())

            progress_bar = st.progress(0.0, text="Đang đọc và xử lý video...")

            def _on_progress(done: int, total: int) -> None:
                ratio = min(1.0, done / max(1, total))
                progress_bar.progress(ratio, text=f"Đã xử lý {done}/{total} frames...")

            processor = VideoPrivacyProcessor(
                pipeline=pipeline,
                privacy_filter=PrivacyFilter(filter_cfg),
            )
            vcfg = VideoConfig(
                frame_stride=int(frame_stride),
                max_frames=int(max_frames_val) if max_frames_val > 0 else None,
            )
            result = processor.process_video(
                in_path,
                out_path,
                filter_config=filter_cfg,
                video_config=vcfg,
                progress_callback=_on_progress,
            )
            progress_bar.progress(1.0, text="Hoàn tất xử lý video!")

            st.success(
                f"Xử lý xong {result.written_frames} frames trong {result.elapsed_sec:.2f}s "
                f"(FPS gốc: {result.input_fps:.2f} → FPS xuất: {result.output_fps:.2f})."
            )
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Input FPS", f"{result.input_fps:.2f}")
            m2.metric("Output FPS", f"{result.output_fps:.2f}")
            m3.metric("Độ phân giải", f"{result.width}x{result.height}")
            m4.metric("Latency TB/frame", f"{result.avg_inference_ms:.1f} ms")

            out_bytes = out_path.read_bytes()
            st.video(out_bytes)
            st.download_button(
                label="Tải video kết quả (.mp4)",
                data=out_bytes,
                file_name=out_path.name,
                mime="video/mp4",
                use_container_width=True,
            )


def main() -> None:
    st.set_page_config(
        page_title="Face Detection & Segmentation — Team 5",
        page_icon="🎭",
        layout="wide",
    )
    st.title("Face Detection & Segmentation — Privacy Filter (Team 5)")
    st.caption("Pipeline tích hợp: Detection (RetinaFace) → Crop → U-Net → Map-back → Mask/Blur")

    pipeline, filter_cfg = _render_sidebar()
    tab_img, tab_vid = st.tabs(["Xử lý ảnh (P0)", "Xử lý video ngắn (P1)"])
    with tab_img:
        _render_image_tab(pipeline, filter_cfg)
    with tab_vid:
        _render_video_tab(pipeline, filter_cfg)


if __name__ == "__main__":
    main()
