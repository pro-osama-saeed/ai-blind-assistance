import io
from collections import Counter

import cv2
import numpy as np
import streamlit as st
from gtts import gTTS
from ultralytics import YOLO

# ── Page config (must be first Streamlit call) ────────────────────────────────

st.set_page_config(
    page_title="Blind Assistance: Environment Scanner",
    page_icon="👁️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Configuration ─────────────────────────────────────────────────────────────

# Classes treated as hazards when Safety mode is enabled.
# Adjust this set to add/remove hazard categories.
HAZARD_CLASSES = {
    "person", "car", "bus", "truck", "motorcycle", "bicycle",
    "dog", "cat", "horse", "cow",
}

# Area-ratio thresholds for distance buckets (box_area / image_area).
NEAR_THRESHOLD = 0.10   # >= 10% of frame → near
MEDIUM_THRESHOLD = 0.03  # >= 3% of frame → medium; below → far

# Maximum number of non-hazard objects to include in the spoken summary.
MAX_OTHER_OBJECTS = 3

# Supported gTTS languages (subset of most common ones).
TTS_LANGUAGES = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Arabic": "ar",
    "Chinese (Mandarin)": "zh-CN",
    "Japanese": "ja",
    "Korean": "ko",
}


# ── Model ─────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    """Load YOLOv8n once and cache it across Streamlit reruns."""
    return YOLO("yolov8n.pt")


# ── Detection helpers ─────────────────────────────────────────────────────────

def get_position(cx: float, img_w: int) -> str:
    """Return 'left', 'center', or 'right' based on bounding-box center x."""
    ratio = cx / img_w
    if ratio < 0.33:
        return "left"
    if ratio > 0.67:
        return "right"
    return "center"


def get_distance(box_area: float, img_area: int) -> str:
    """Return 'near', 'medium', or 'far' based on the box-area fraction."""
    ratio = box_area / img_area
    if ratio >= NEAR_THRESHOLD:
        return "near"
    if ratio >= MEDIUM_THRESHOLD:
        return "medium"
    return "far"


def parse_detections(results, conf_threshold: float, img_w: int, img_h: int) -> list:
    """
    Extract per-detection metadata from YOLOv8 results.

    Returns a list of dicts with keys:
        name, conf, position, distance, is_hazard, area_ratio
    Only boxes with confidence >= conf_threshold are included.
    """
    img_area = img_w * img_h
    detections = []
    for box in results[0].boxes:
        conf = float(box.conf[0])
        if conf < conf_threshold:
            continue
        class_id = int(box.cls[0])
        name = results[0].names[class_id]
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx = (x1 + x2) / 2
        box_area = (x2 - x1) * (y2 - y1)
        detections.append({
            "name": name,
            "conf": conf,
            "position": get_position(cx, img_w),
            "distance": get_distance(box_area, img_area),
            "is_hazard": name in HAZARD_CLASSES,
            "area_ratio": box_area / img_area,
        })
    return detections


def summarize_detections(detections: list, hazard_mode: bool) -> str:
    """
    Build a natural spoken sentence from the detection list.

    Priority order:
      1. Hazard objects (announced with 'Warning:' when hazard_mode is on).
      2. Up to MAX_OTHER_OBJECTS non-hazard objects sorted by proximity (largest box first).
    """
    if not detections:
        return "No objects detected in the current view."

    hazards = [d for d in detections if d["is_hazard"]]
    others = sorted(
        [d for d in detections if not d["is_hazard"]],
        key=lambda d: d["area_ratio"],
        reverse=True,
    )[:MAX_OTHER_OBJECTS]

    parts = []
    for d in hazards:
        desc = f"{d['name']} on the {d['position']}, {d['distance']}"
        parts.append(f"Warning: {desc}" if hazard_mode else desc)
    for d in others:
        parts.append(f"{d['name']} on the {d['position']}, {d['distance']}")

    return "I see " + "; ".join(parts) + "."


def generate_tts_audio(text: str, lang: str = "en", slow: bool = False) -> bytes:
    """Generate MP3 audio for the given text using gTTS.

    Audio is produced entirely in memory — no files are written to disk.

    Returns:
        MP3-encoded audio as raw bytes, ready to pass to st.audio().
    Raises:
        RuntimeError: if gTTS generation fails.
    """
    try:
        buf = io.BytesIO()
        gTTS(text=text, lang=lang, slow=slow).write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except Exception as exc:
        raise RuntimeError(f"Audio generation failed: {exc}") from exc


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Controls")

    st.subheader("🔍 Detection settings")
    hazard_mode = st.checkbox(
        "🚨 Safety / hazard mode",
        value=True,
        help="Prefix detected hazards (vehicles, animals, people) with 'Warning:' in the voice output.",
    )
    conf_threshold = st.slider(
        "Confidence threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
        help="Only detections above this confidence score are included. Lower = more detections; higher = fewer but more certain.",
    )

    st.divider()
    st.subheader("🔊 Audio controls")
    lang_label = st.selectbox(
        "Voice language",
        options=list(TTS_LANGUAGES.keys()),
        index=0,
        help="Language used for the spoken description.",
    )
    tts_lang = TTS_LANGUAGES[lang_label]
    slow_speech = st.checkbox(
        "Slow speech",
        value=False,
        help="Read the description at a slower pace (useful for language learners or difficult-to-hear environments).",
    )

    st.divider()
    st.subheader("♿ Accessibility")
    screen_reader_mode = st.checkbox(
        "Screen-reader friendly mode",
        value=False,
        help="Hides the camera preview and annotated image by default; emphasises text output. Useful for screen-reader users.",
    )

    st.divider()
    with st.expander("🔒 Privacy note"):
        st.write(
            "All image processing happens **locally on this machine** using the YOLOv8 model. "
            "No images or detection results are transmitted to any external server. "
            "Captured frames are not stored to disk."
        )


# ── Main UI ───────────────────────────────────────────────────────────────────

st.title("👁️ Blind Assistance: Environment Scanner")
st.caption(
    "Capture an image to scan your surroundings. "
    "The AI will detect objects and describe them with location and distance."
)

# Step indicator
st.markdown("**How it works:** 📷 Capture → 🔍 Scan → 📋 Results → 🔊 Voice")
st.divider()

# ── Session state initialisation ──────────────────────────────────────────────
if "last_sentence" not in st.session_state:
    st.session_state.last_sentence = None
if "last_audio" not in st.session_state:
    st.session_state.last_audio = None
if "last_detections" not in st.session_state:
    st.session_state.last_detections = None

# ── Model loading ─────────────────────────────────────────────────────────────
try:
    model = load_model()
except Exception as exc:
    st.error(
        f"⚠️ Failed to load the detection model: {exc}\n\n"
        "Ensure `yolov8n.pt` is available and the `ultralytics` package is installed."
    )
    st.stop()

# ── Camera step ───────────────────────────────────────────────────────────────
st.subheader("Step 1 — Capture")
if not screen_reader_mode:
    camera_image = st.camera_input(
        "Point your camera at the scene and press the capture button",
        help="A new capture automatically triggers a fresh scan.",
    )
else:
    st.info(
        "📷 Screen-reader mode: the camera preview is hidden. "
        "Use the file uploader below to provide an image."
    )
    uploaded = st.file_uploader(
        "Upload an image to scan",
        type=["jpg", "jpeg", "png"],
        help="Upload a photo to scan when camera preview is hidden.",
    )
    camera_image = uploaded  # unify the variable name for the rest of the flow

# ── Processing ────────────────────────────────────────────────────────────────
if camera_image is not None:
    st.subheader("Step 2 — Scanning")

    with st.spinner("Running YOLOv8 object detection…"):
        try:
            bytes_data = camera_image.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            if cv2_img is None:
                st.error("⚠️ Could not decode the image. Please try capturing again.")
                st.stop()
            img_h, img_w = cv2_img.shape[:2]

            results = model(cv2_img, verbose=False)
            detections = parse_detections(results, conf_threshold, img_w, img_h)
            annotated_frame = results[0].plot()
        except Exception as exc:
            st.error(
                f"⚠️ Object detection failed: {exc}\n\n"
                "Try recapturing the image or lowering the confidence threshold."
            )
            st.stop()

    st.toast("✅ Scan complete!", icon="✅")

    # Persist results in session state for replay
    spoken_sentence = summarize_detections(detections, hazard_mode)
    st.session_state.last_sentence = spoken_sentence
    st.session_state.last_detections = detections

    try:
        audio_bytes = generate_tts_audio(spoken_sentence, lang=tts_lang, slow=slow_speech)
        st.session_state.last_audio = audio_bytes
    except RuntimeError as exc:
        st.warning(f"🔇 {exc}  —  Text output is still available below.")
        st.session_state.last_audio = None

# ── Results display ───────────────────────────────────────────────────────────
if st.session_state.last_sentence is not None:
    st.subheader("Step 3 — Results")

    detections = st.session_state.last_detections or []
    spoken_sentence = st.session_state.last_sentence

    tab_text, tab_visual, tab_details = st.tabs(["📋 Text summary", "🖼️ Visual", "🔢 Details"])

    with tab_text:
        if detections:
            hazard_items = [d for d in detections if d["is_hazard"]]
            if hazard_items and hazard_mode:
                st.warning(
                    "⚠️ **Hazards detected:** "
                    + ", ".join(
                        f"{d['name']} ({d['position']}, {d['distance']})"
                        for d in hazard_items
                    )
                )
            st.success(f"**Voice output:** {spoken_sentence}")
        else:
            st.info("ℹ️ No objects were detected. Try adjusting the confidence threshold or recapturing.")

    with tab_visual:
        if screen_reader_mode:
            st.info("🖼️ Annotated image hidden in screen-reader mode.")
        elif "annotated_frame" in dir() and annotated_frame is not None:
            st.image(annotated_frame, channels="BGR", caption="AI Detection Results", use_container_width=True)
        else:
            st.info("Capture a new image to see the annotated visual.")

    with tab_details:
        if detections:
            counts = Counter(d["name"] for d in detections)
            st.markdown("**Objects detected (by class):**")
            for cls_name, cnt in sorted(counts.items(), key=lambda x: -x[1]):
                is_hazard = cls_name in HAZARD_CLASSES
                badge = " 🚨" if (is_hazard and hazard_mode) else ""
                st.write(f"- **{cls_name}**{badge}: {cnt} instance{'s' if cnt > 1 else ''}")

            st.markdown("**All detections (sorted by size):**")
            sorted_dets = sorted(detections, key=lambda d: d["area_ratio"], reverse=True)
            for i, d in enumerate(sorted_dets, 1):
                conf_pct = f"{d['conf'] * 100:.0f}%"
                hazard_tag = " 🚨 hazard" if (d["is_hazard"] and hazard_mode) else ""
                st.write(
                    f"{i}. **{d['name']}**{hazard_tag} — "
                    f"{d['position']}, {d['distance']} "
                    f"(conf: {conf_pct})"
                )
        else:
            st.info("No detections to display.")

    # ── Audio playback ────────────────────────────────────────────────────────
    st.subheader("Step 4 — Voice output")

    if st.session_state.last_audio is not None:
        st.audio(st.session_state.last_audio, format="audio/mp3", autoplay=True)
        if st.button("🔁 Replay audio", help="Play the spoken description again without recapturing."):
            st.audio(st.session_state.last_audio, format="audio/mp3", autoplay=True)
    else:
        st.warning("Audio is unavailable. Please check your network connection and try again.")
