import io

import cv2
import numpy as np
import streamlit as st
from gtts import gTTS
from ultralytics import YOLO

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


def generate_tts_audio(text: str) -> bytes:
    """Generate MP3 audio for the given text using gTTS.

    Audio is produced entirely in memory — no files are written to disk.

    Returns:
        MP3-encoded audio as raw bytes, ready to pass to st.audio().
    """
    buf = io.BytesIO()
    gTTS(text=text, lang="en", slow=False).write_to_fp(buf)
    buf.seek(0)
    return buf.read()


# ── Streamlit UI ──────────────────────────────────────────────────────────────

st.title("Blind Assistance: Environment Scanner")
st.write(
    "Capture an image to scan your surroundings. "
    "The AI will detect objects and describe them with location and distance."
)

# Controls
hazard_mode = st.checkbox("🚨 Safety / hazard mode", value=True)
conf_threshold = st.slider(
    "Confidence threshold", min_value=0.10, max_value=0.90, value=0.25, step=0.05
)

model = load_model()

# Camera input — a new capture triggers a full rerun automatically
camera_image = st.camera_input("Capture and Scan Environment")

if camera_image is not None:
    st.write("Scanning image with YOLOv8…")

    # Decode the captured image into an OpenCV array
    bytes_data = camera_image.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    img_h, img_w = cv2_img.shape[:2]

    # Run YOLOv8 inference
    results = model(cv2_img, verbose=False)

    # Filter and enrich detections with spatial/distance metadata
    detections = parse_detections(results, conf_threshold, img_w, img_h)

    # Show the annotated frame (Streamlit needs RGB, OpenCV gives BGR)
    annotated_frame = results[0].plot()
    st.image(annotated_frame, channels="BGR", caption="AI Detection Results")

    # Build the spoken sentence and play audio (in-memory, no MP3 files saved)
    spoken_sentence = summarize_detections(detections, hazard_mode)
    st.success(f"**Voice Output:** {spoken_sentence}")
    audio_bytes = generate_tts_audio(spoken_sentence)
    st.audio(audio_bytes, format="audio/mp3", autoplay=True)
