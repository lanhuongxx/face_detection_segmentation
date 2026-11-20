"""FaceShield: giao diện Streamlit.
Chạy:  streamlit run app.py        (từ thư mục gốc project, đã bật môi trường .venv)

Giữ lại từ bản app.py cũ: cache pipeline theo thời điểm sửa file trọng số, Lock khi chạy model,
nhận diện ảnh mới bằng sha256, bắt lỗi file. Bổ sung: che mặt, thẻ khuôn mặt, tải về, tab camera/video."""
import json
import tempfile
from hashlib import sha256
from pathlib import Path
from threading import Lock

import streamlit as st

from src.pipeline.inference import FacePipeline
from src.pipeline.preprocess_input import CorruptImage, FileTooLarge, UnsupportedFormat, load_image
from src.pipeline.privacy_filter import covered_fraction
from src.utils.config import load_config
from src.utils.io import extract_frame, to_png_bytes, video_info

ROOT = Path(__file__).resolve().parent
MODES = {"Làm mờ": "blur", "Pixel hóa": "pixelate", "Tô màu": "solid"}
REGIONS = {"Theo khuôn mặt (mask)": "mask", "Hình chữ nhật (box)": "bbox"}
IMAGE_TYPES = ["jpg", "jpeg", "png", "webp"]
VIDEO_TYPES = ["mp4", "mov", "avi", "mkv"]
MAX_CARDS = 60          # số thẻ khuôn mặt hiển thị tối đa (ảnh đông người có thể có hàng trăm mặt)
# Đề không yêu cầu che mặt nên tạm TẮT. Đặt True để bật: kiểu che, chọn mặt cần che, tab Kết quả đã che, tab So sánh.
ENABLE_PRIVACY = False


# ---------------------------------------------------------------- pipeline
@st.cache_resource
def get_pipeline(weights_path, weights_modified):
    """Nạp pipeline một lần. weights_modified làm cache tự đổi khi bạn train lại model."""
    config = load_config(ROOT / "configs" / "pipeline.yaml")
    config["segmentation"]["weights"] = weights_path
    return FacePipeline(cfg=config), Lock()


def load_pipeline():
    """Trả về (config, (pipeline, lock)). Báo lỗi nếu chưa có checkpoint."""
    config = load_config(ROOT / "configs" / "pipeline.yaml")
    weights = ROOT / config["segmentation"]["weights"]
    if not weights.is_file():
        raise FileNotFoundError(f"Thiếu checkpoint: {weights}. Chạy CELL 13 trong notebook để lưu model.")
    return config, get_pipeline(str(weights), weights.stat().st_mtime_ns)


# ---------------------------------------------------------------- trạng thái
def clear_selection(key):
    for name in [n for n in st.session_state if str(n).startswith(f"{key}_sel_")]:
        del st.session_state[name]


def set_all(key, ids, value):
    for face_id in ids:
        st.session_state[f"{key}_sel_{face_id}"] = value


def selected_ids(key, faces):
    for face in faces:
        st.session_state.setdefault(f"{key}_sel_{face.id}", True)     # mặc định che tất cả
    return [f.id for f in faces if st.session_state[f"{key}_sel_{f.id}"]]


def run_analysis(key, image, source, image_id=None):
    """Chạy phát hiện + phân vùng và lưu kết quả vào session_state[key]."""
    st.session_state.pop(key, None)
    clear_selection(key)
    try:
        config, (pipeline, lock) = load_pipeline()
        with st.spinner("Đang phát hiện và phân vùng khuôn mặt…"):
            with lock:
                result = pipeline.analyze(image, conf_threshold=conf, source=source,
                                          mask_threshold=config["segmentation"]["mask_threshold"])
        st.session_state[key] = {"id": image_id, "result": result}
        st.session_state["last"] = key
    except FileNotFoundError as error:
        st.error(str(error))
    except Exception as error:
        st.error(f"Không thể chạy pipeline: {error}")


# ---------------------------------------------------------------- hiển thị kết quả
def render_result(key):
    entry = st.session_state.get(key)
    if entry is None:
        return
    result = entry["result"]
    faces = result.faces
    st.divider()
    if not faces:
        st.info("Không phát hiện khuôn mặt nào. Thử hạ **Ngưỡng phát hiện** ở thanh bên rồi phân tích lại.")
        st.image(result.image)
        return
    try:
        _, (pipeline, _) = load_pipeline()
    except FileNotFoundError as error:
        st.error(str(error))
        return

    ids = [f.id for f in faces]
    if ENABLE_PRIVACY:
        selected = selected_ids(key, faces)
        final = pipeline.apply_privacy(result, selected, mode=MODES[mode_label], strength=strength,
                                       region=REGIONS[region_label])
    else:
        final = None
        for face in faces:
            face.blurred = False            # không che gì, báo cáo JSON phải ghi đúng như vậy

    c1, c2, c3 = st.columns(3)
    c1.metric("Số khuôn mặt", len(faces))
    if ENABLE_PRIVACY:
        c2.metric("Đang che", len(selected))
    else:
        c2.metric("Phân vùng bằng U-Net", sum(f.status == "ok" for f in faces))
    c3.metric("Thời gian phân tích", f"{result.timing_ms['total']:.0f} ms")

    n_small = sum(f.reason == "too_small" for f in faces)
    n_bad = sum(f.reason == "bad_mask" for f in faces)
    if n_small or n_bad:
        st.info(f"{n_small} mặt quá nhỏ (< {pipeline.cfg['crop']['min_face_px']} px) và {n_bad} mặt có mask bất thường "
                "được thay bằng hình elip nội tiếp trong khung phát hiện. Ảnh có nhiều mặt nhỏ thì vẫn có thể còn sót "
                "mặt chưa phát hiện, hãy kiểm tra tab Phát hiện.")

    labels = ["Ảnh gốc", "Phát hiện", "Phân vùng"] + (["Kết quả (đã che)"] if ENABLE_PRIVACY else [])
    result_tabs = st.tabs(labels)
    result_tabs[0].image(result.image)
    result_tabs[1].image(result.overlay_boxes)
    result_tabs[2].image(result.overlay_masks)
    if ENABLE_PRIVACY:
        result_tabs[3].image(final)

    with st.expander(f"Thẻ khuôn mặt ({len(faces)})", expanded=len(faces) <= 24):
        if ENABLE_PRIVACY:
            b1, b2, _ = st.columns([1, 1, 6])
            b1.button("Chọn tất cả", key=f"{key}_all", on_click=set_all, args=(key, ids, True))
            b2.button("Bỏ chọn", key=f"{key}_none", on_click=set_all, args=(key, ids, False))
        shown = faces[:MAX_CARDS]
        per_row = 6
        for start in range(0, len(shown), per_row):
            for col, face in zip(st.columns(per_row), shown[start:start + per_row]):
                with col:
                    st.image(face.thumbnail, width=96)
                    w, h = face.size_px
                    st.caption(f"#{face.id} · score {face.score:.2f}")
                    st.caption(f"{w}×{h} px · {'mask' if face.status == 'ok' else 'elip'}")
                    st.caption("Danh tính: —")
                    if ENABLE_PRIVACY:
                        st.checkbox("Che", key=f"{key}_sel_{face.id}")
        if len(faces) > MAX_CARDS:
            st.caption(f"Chỉ hiển thị {MAX_CARDS} thẻ đầu (danh sách đầy đủ có trong báo cáo JSON).")

    stem = Path(result.meta.get("source", {}).get("filename") or "faceshield").stem
    d1, d2, _ = st.columns([1, 1, 4])
    if ENABLE_PRIVACY:
        d1.download_button("Tải ảnh đã che (PNG)", data=to_png_bytes(final), file_name=f"{stem}_faceshield.png",
                           mime="image/png", key=f"{key}_dl_png")
    else:
        d1.download_button("Tải ảnh phân vùng (PNG)", data=to_png_bytes(result.overlay_masks),
                           file_name=f"{stem}_masks.png", mime="image/png", key=f"{key}_dl_png")
    d2.download_button("Tải báo cáo (JSON)", data=json.dumps(pipeline.report(result), ensure_ascii=False, indent=2),
                       file_name=f"{stem}_report.json", mime="application/json", key=f"{key}_dl_json")


# ---------------------------------------------------------------- thanh bên: tùy chọn
st.set_page_config(page_title="FaceShield · Phát hiện và phân vùng khuôn mặt", layout="wide")
with st.sidebar:
    st.header("Tùy chọn")
    conf = st.slider("Ngưỡng phát hiện", min_value=0.10, max_value=0.90, value=0.50, step=0.05,
                     help="Thấp hơn: ít bỏ sót mặt hơn nhưng có thể thêm mặt nhầm (bỏ chọn ở thẻ khuôn mặt).")
    if ENABLE_PRIVACY:
        mode_label = st.radio("Kiểu che", list(MODES))
        strength = st.slider("Độ mạnh", min_value=1, max_value=10, value=6)
        region_label = st.radio("Vùng che", list(REGIONS))
        st.caption("Ảnh chỉ xử lý trong phiên làm việc, không được lưu lại. Che mặt là công cụ hỗ trợ, "
                   "hãy kiểm tra kết quả trước khi chia sẻ.")
    else:
        mode_label, strength, region_label = list(MODES)[0], 6, list(REGIONS)[0]    # không dùng khi tắt che mặt
        st.caption("Ảnh chỉ xử lý trong phiên làm việc, không được lưu lại.")

st.title("FaceShield · Phát hiện và phân vùng khuôn mặt")
top_tabs = st.tabs(["Ảnh", "Camera / Video"] + (["So sánh Box vs Mask"] if ENABLE_PRIVACY else []))
tab_image, tab_camera = top_tabs[0], top_tabs[1]

# ---------------------------------------------------------------- tab Ảnh
with tab_image:
    upload = st.file_uploader("Tải ảnh JPG, PNG hoặc WebP", type=IMAGE_TYPES, key="image_upload")
    samples = sorted(p for p in (ROOT / "samples").glob("*") if p.suffix.lower().lstrip(".") in IMAGE_TYPES)
    sample_name = st.selectbox("Hoặc dùng ảnh mẫu", ["(không)"] + [p.name for p in samples]) if samples else "(không)"

    data, filename = None, None
    if upload is not None:
        data, filename = upload.getvalue(), upload.name
    elif sample_name != "(không)":
        data, filename = (ROOT / "samples" / sample_name).read_bytes(), sample_name

    if data is None:
        st.session_state.pop("img", None)
        st.info("Hãy tải một ảnh lên hoặc chọn ảnh mẫu.")
    else:
        image_id = sha256(data).hexdigest()
        current = st.session_state.get("img")
        if current is not None and current["id"] != image_id:       # đổi sang ảnh khác: bỏ kết quả cũ
            st.session_state.pop("img")
            clear_selection("img")
        if st.button("Phân tích", type="primary", key="analyze_image"):
            try:
                max_mb = load_config(ROOT / "configs" / "pipeline.yaml")["preprocess"]["max_file_mb"]
                image, meta = load_image(data, max_file_mb=max_mb)
                run_analysis("img", image, {**meta, "type": "image", "filename": filename}, image_id)
            except UnsupportedFormat:
                st.error("Vui lòng tải ảnh JPG, PNG hoặc WebP.")
            except CorruptImage:
                st.error("Ảnh bị lỗi hoặc không đọc được. Hãy chọn ảnh khác.")
            except FileTooLarge:
                st.error("Ảnh vượt quá dung lượng cho phép trong cấu hình pipeline.")
        render_result("img")

# ---------------------------------------------------------------- tab Camera / Video
with tab_camera:
    st.caption("Chọn một khung hình từ video camera rồi phân tích như ảnh tĩnh. "
               "Xử lý cả video (từng khung hình) chưa có trong bản này.")
    video = st.file_uploader("Video camera (MP4, MOV, AVI, MKV)", type=VIDEO_TYPES, key="video_upload")
    if video is None:
        st.session_state.pop("cam", None)
        st.session_state.pop("video_id", None)
    else:
        video_key = f"{video.name}_{video.size}"
        if st.session_state.get("video_id") != video_key:           # video mới: ghi ra file tạm
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(video.name).suffix)
            tmp.write(video.getvalue())
            tmp.close()
            st.session_state["video_id"] = video_key
            st.session_state["video_path"] = tmp.name
            st.session_state.pop("cam", None)
            clear_selection("cam")
        path = st.session_state["video_path"]
        try:
            info = video_info(path)
            total = max(info["total_frames"], 1)
            st.caption(f"{info['width']}×{info['height']} · {info['fps']:.0f} FPS · "
                       f"{info['duration_s']:.1f} giây · {info['total_frames']} khung hình")
            frame_idx = st.slider("Khung hình", 0, total - 1, 0) if total > 1 else 0
            frame = extract_frame(path, frame_idx)
            st.image(frame, caption=f"Khung hình {frame_idx}")
            if st.button("Phân tích khung hình này", type="primary", key="analyze_frame"):
                run_analysis("cam", frame, {"type": "video_frame", "filename": video.name, "frame_idx": frame_idx},
                             f"{video_key}_{frame_idx}")
        except ValueError as error:
            st.error(str(error))
        render_result("cam")

# ---------------------------------------------------------------- tab So sánh
if ENABLE_PRIVACY:
    tab_compare = top_tabs[2]
    with tab_compare:
        last = st.session_state.get("last")
        entry = st.session_state.get(last) if last else None
        if entry is None or not entry["result"].faces:
            st.info("Hãy phân tích một ảnh (hoặc khung hình) có khuôn mặt ở tab Ảnh hoặc Camera / Video trước.")
        else:
            result = entry["result"]
            try:
                _, (pipeline, _) = load_pipeline()
                chosen = selected_ids(last, result.faces)
                st.caption("Cùng kiểu che và độ mạnh ở thanh bên; chỉ khác vùng che. Mask che đúng hình dạng mặt, "
                           "box che cả hình chữ nhật nên phá nhiều nền hơn.")
                left, right = st.columns(2)
                for column, region, title in ((left, "bbox", "Che theo BOX (hình chữ nhật)"),
                                              (right, "mask", "Che theo MASK (U-Net)")):
                    image = pipeline.apply_privacy(result, chosen, mode=MODES[mode_label], strength=strength,
                                                   region=region, store=False)
                    fraction = covered_fraction(result.image.shape, result.faces, set(chosen), region)
                    column.subheader(title)
                    column.image(image)
                    column.metric("Diện tích ảnh bị che", f"{fraction:.1%}")
            except FileNotFoundError as error:
                st.error(str(error))
