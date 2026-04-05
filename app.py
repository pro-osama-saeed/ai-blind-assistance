import base64
import io
import json
import queue
import threading
import time
from collections import Counter
from datetime import datetime, timezone

import av
import cv2
import numpy as np
import streamlit as st
from gtts import gTTS
from ultralytics import YOLO

# ── Optional dependencies ─────────────────────────────────────────────────────

try:
    from streamlit_webrtc import VideoProcessorBase, webrtc_streamer, WebRtcMode

    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

try:
    import gspread
    from google.oauth2.service_account import Credentials

    GSPREAD_AVAILABLE = True
    _WorksheetNotFound = gspread.WorksheetNotFound
except ImportError:
    GSPREAD_AVAILABLE = False
    _WorksheetNotFound = Exception  # never reached when gspread absent

# ── Page config (must be first Streamlit call) ────────────────────────────────

st.set_page_config(
    page_title="Blind Assistance: Environment Scanner",
    page_icon="👁️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Configuration ─────────────────────────────────────────────────────────────

APP_VERSION = "0.2.0"

# Classes treated as hazards when Safety mode is enabled.
HAZARD_CLASSES = {
    "person", "car", "bus", "truck", "motorcycle", "bicycle",
    "dog", "cat", "horse", "cow",
}

# Area-ratio thresholds for distance buckets (box_area / image_area).
NEAR_THRESHOLD = 0.10    # >= 10% of frame → near
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

FEEDBACK_CATEGORIES = [
    "Wrong label",
    "Missed object",
    "Wrong position",
    "Distance misleading",
    "Audio issue",
    "UI issue",
    "Other",
]

# Live-mode parameters
LIVE_SPEECH_INTERVAL = 5   # minimum seconds between spoken alerts
LIVE_PROCESS_EVERY_N = 3   # run YOLO every Nth frame (reduces CPU load)
LIVE_RESIZE_WIDTH = 640    # downscale frames to this width before inference
RTC_CONFIGURATION = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
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
    """Extract per-detection metadata from YOLOv8 results."""
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
    """Build a natural spoken sentence from the detection list."""
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


def build_grouped_summary(detections: list, hazard_mode: bool) -> dict:
    """
    Group detections by position (left / center / right) and deduplicate
    classes within each group.  Returns a dict keyed by position.
    """
    groups: dict = {"left": [], "center": [], "right": []}
    seen: dict = {"left": set(), "center": set(), "right": set()}
    for d in sorted(detections, key=lambda x: x["area_ratio"], reverse=True):
        pos = d["position"]
        key = (d["name"], d["distance"])
        if key not in seen[pos]:
            seen[pos].add(key)
            groups[pos].append(d)
    return groups


# ── TTS helper ────────────────────────────────────────────────────────────────


def generate_tts_audio(text: str, lang: str = "en", slow: bool = False) -> bytes:
    """Generate MP3 audio for the given text using gTTS (fully in-memory)."""
    try:
        buf = io.BytesIO()
        gTTS(text=text, lang=lang, slow=slow).write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except Exception as exc:
        raise RuntimeError(f"Audio generation failed: {exc}") from exc


# ── Google Sheets helpers ─────────────────────────────────────────────────────


def _get_sheets_client():
    """Return (gspread_client, sheet_id, error_msg)."""
    if not GSPREAD_AVAILABLE:
        return None, None, (
            "gspread / google-auth are not installed. "
            "Run: pip install gspread google-auth"
        )
    try:
        creds_dict = dict(st.secrets["gcp_service_account"])
        sheet_id = str(st.secrets["google_sheet_id"])
    except KeyError:
        return None, None, (
            "Google Sheets secrets not found in .streamlit/secrets.toml. "
            "Add [gcp_service_account] and google_sheet_id."
        )
    try:
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        client = gspread.authorize(creds)
        return client, sheet_id, None
    except Exception as exc:
        return None, None, f"Sheets authentication error: {exc}"


def check_sheets_config() -> tuple:
    """Return (available: bool, error_msg: str | None)."""
    _, _, err = _get_sheets_client()
    return (err is None), err


_FEEDBACK_COLUMNS = [
    "timestamp", "app_version", "category", "comment",
    "user_contact", "hazard_mode", "conf_threshold",
    "language_label", "language_code", "slow_speech",
    "screen_reader_mode", "spoken_sentence",
    "detection_summary", "class_counts", "image_b64",
]


def append_feedback_to_sheet(row_data: dict) -> tuple:
    """Append one feedback row. Returns (success: bool, error_msg | None)."""
    client, sheet_id, err = _get_sheets_client()
    if err:
        return False, err
    try:
        sh = client.open_by_key(sheet_id)
        try:
            ws = sh.worksheet("Feedback")
        except _WorksheetNotFound:
            ws = sh.add_worksheet(title="Feedback", rows=1000, cols=len(_FEEDBACK_COLUMNS))
            ws.append_row(_FEEDBACK_COLUMNS)
        ws.append_row([row_data.get(k, "") for k in _FEEDBACK_COLUMNS])
        return True, None
    except Exception as exc:
        return False, str(exc)


# ── Image encoding helper ─────────────────────────────────────────────────────


def encode_image_for_sheet(image_bytes: bytes, max_kb: int = 150) -> str:
    """
    Compress *image_bytes* to a small JPEG and return a base64 string.
    Falls back to 'too_large' or an error marker if the result is still big.
    """
    max_size = max_kb * 1024
    try:
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return "decode_error"
        h, w = img.shape[:2]
        if w > 320:
            img = cv2.resize(img, (320, int(h * 320 / w)))
        for quality in [80, 50, 30, 10]:
            ok, buf = cv2.imencode(
                ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality]
            )
            if ok:
                b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
                if len(b64.encode("utf-8")) <= max_size:
                    return b64
        return "too_large"
    except Exception as exc:
        return f"encode_error:{type(exc).__name__}"


# ── Live-video processor ──────────────────────────────────────────────────────

if WEBRTC_AVAILABLE:
    class YOLOVideoProcessor(VideoProcessorBase):
        """Runs YOLO on every Nth frame and overlays bounding boxes."""

        def __init__(self):
            self.model = load_model()
            self.conf_threshold: float = 0.25
            self.hazard_mode: bool = True
            self._lock = threading.Lock()
            self._frame_count: int = 0
            self.last_detections: list = []
            self._last_annotated = None
            self.result_queue: queue.Queue = queue.Queue(maxsize=5)

        def get_detections(self) -> list:
            """Return a thread-safe snapshot of the latest detections."""
            with self._lock:
                return list(self.last_detections)

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            self._frame_count += 1
            img = frame.to_ndarray(format="bgr24")
            h, w = img.shape[:2]

            if self._frame_count % LIVE_PROCESS_EVERY_N == 0:
                new_w = min(LIVE_RESIZE_WIDTH, w)
                new_h = int(h * new_w / w)
                small = cv2.resize(img, (new_w, new_h))
                sh, sw = small.shape[:2]

                results = self.model(small, verbose=False)
                dets = parse_detections(
                    results, self.conf_threshold, sw, sh
                )
                annotated_small = results[0].plot()
                annotated = cv2.resize(annotated_small, (w, h))

                with self._lock:
                    self.last_detections = dets
                    self._last_annotated = annotated

                try:
                    self.result_queue.put_nowait(dets)
                except queue.Full:
                    pass

            with self._lock:
                out = (
                    self._last_annotated
                    if self._last_annotated is not None
                    else img
                )
            return av.VideoFrame.from_ndarray(out, format="bgr24")


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Controls")

    app_mode = st.radio(
        "📷 Mode",
        options=["Single capture", "Live video"],
        index=0,
        help=(
            "**Single capture**: take one photo and scan it.\n\n"
            "**Live video**: continuous detection from your camera."
        ),
    )

    st.divider()

    with st.expander("🔍 Detection settings", expanded=True):
        hazard_mode = st.checkbox(
            "🚨 Safety / hazard mode",
            value=True,
            help="Prefix detected hazards with 'Warning:' in voice output.",
        )
        conf_threshold = st.slider(
            "Confidence threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.25,
            step=0.05,
            help="Lower = more detections; higher = fewer but more certain.",
        )

    with st.expander("🔊 Audio controls", expanded=False):
        autoplay = st.checkbox(
            "Autoplay audio after scan",
            value=True,
            help="Automatically play voice output after each new scan.",
        )
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
            help="Read the description at a slower pace.",
        )
        if app_mode == "Live video":
            speak_alerts_live = st.checkbox(
                "🔊 Auto-speak alerts (live)",
                value=False,
                help=f"Announce detections every {LIVE_SPEECH_INTERVAL} s while live mode is active.",
            )
        else:
            speak_alerts_live = False

    with st.expander("♿ Accessibility", expanded=False):
        screen_reader_mode = st.checkbox(
            "Screen-reader friendly mode",
            value=False,
            help=(
                "Hides camera preview and annotated image; "
                "emphasises text output. Useful for screen-reader users."
            ),
        )

    st.divider()
    with st.expander("🔒 Privacy note"):
        st.write(
            "**Image processing** is performed **locally** using YOLOv8. "
            "No pixel data is sent to external servers.\n\n"
            "**Voice synthesis** uses Google TTS, which sends the generated "
            "text description to Google's servers.\n\n"
            "**Feedback**: images are only uploaded if you explicitly check "
            "the *include image* checkbox in the feedback form."
        )


# ── Main UI ───────────────────────────────────────────────────────────────────

st.title("👁️ Blind Assistance: Environment Scanner")
st.caption(
    "Scan your surroundings — the AI detects objects and describes them "
    "with location and approximate distance."
)
st.divider()

# ── Session state initialisation ──────────────────────────────────────────────

for _key in (
    "last_sentence", "last_audio", "last_detections",
    "last_annotated_frame", "last_image_bytes",
):
    if _key not in st.session_state:
        st.session_state[_key] = None

# ── Model loading ─────────────────────────────────────────────────────────────

try:
    model = load_model()
except Exception as exc:
    st.error(
        f"⚠️ Failed to load the detection model: {exc}\n\n"
        "Ensure `yolov8n.pt` is available and `ultralytics` is installed."
    )
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# SINGLE-CAPTURE MODE
# ══════════════════════════════════════════════════════════════════════════════

if app_mode == "Single capture":
    st.markdown("**How it works:** 📷 Capture → 🔍 Scan → 📋 Results → 🔊 Voice")

    # ── Camera step ───────────────────────────────────────────────────────────
    st.subheader("Step 1 — Capture")
    if not screen_reader_mode:
        camera_image = st.camera_input(
            "Point your camera at the scene and press the capture button",
            help="A new capture automatically triggers a fresh scan.",
        )
    else:
        st.info(
            "📷 Screen-reader mode: camera preview is hidden. "
            "Use the file uploader below to provide an image."
        )
        uploaded = st.file_uploader(
            "Upload an image to scan",
            type=["jpg", "jpeg", "png"],
            help="Upload a photo to scan when camera preview is hidden.",
        )
        camera_image = uploaded

    # ── Processing ────────────────────────────────────────────────────────────
    if camera_image is not None:
        st.subheader("Step 2 — Scanning")

        with st.spinner("Running YOLOv8 object detection…"):
            try:
                bytes_data = camera_image.getvalue()
                cv2_img = cv2.imdecode(
                    np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR
                )
                if cv2_img is None:
                    st.error("⚠️ Could not decode the image. Please try again.")
                    st.stop()
                img_h, img_w = cv2_img.shape[:2]

                results = model(cv2_img, verbose=False)
                detections = parse_detections(results, conf_threshold, img_w, img_h)
                annotated_frame = results[0].plot()
            except Exception as exc:
                st.error(
                    f"⚠️ Object detection failed: {exc}\n\n"
                    "Try recapturing or lowering the confidence threshold."
                )
                st.stop()

        st.toast("✅ Scan complete!", icon="✅")

        spoken_sentence = summarize_detections(detections, hazard_mode)
        st.session_state.last_sentence = spoken_sentence
        st.session_state.last_detections = detections
        st.session_state.last_annotated_frame = annotated_frame
        st.session_state.last_image_bytes = bytes_data

        try:
            audio_bytes = generate_tts_audio(
                spoken_sentence, lang=tts_lang, slow=slow_speech
            )
            st.session_state.last_audio = audio_bytes
        except RuntimeError as exc:
            st.warning(f"🔇 {exc}  —  Text output is still available below.")
            st.session_state.last_audio = None

    # ── Results display ───────────────────────────────────────────────────────
    if st.session_state.last_sentence is not None:
        st.subheader("Step 3 — Results")

        detections = st.session_state.last_detections or []
        spoken_sentence = st.session_state.last_sentence

        tab_text, tab_visual, tab_details = st.tabs(
            ["📋 Summary", "🖼️ Visual", "🔢 Details"]
        )

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

                # Grouped / deduplicated view
                groups = build_grouped_summary(detections, hazard_mode)
                st.markdown("---")
                st.markdown("**Detections by position:**")
                pos_icons = {"left": "⬅️", "center": "⬆️", "right": "➡️"}
                for pos in ("left", "center", "right"):
                    items = groups[pos]
                    if not items:
                        continue
                    labels = [
                        f"{'🚨 ' if d['is_hazard'] and hazard_mode else ''}"
                        f"**{d['name']}** ({d['distance']})"
                        for d in items
                    ]
                    st.write(
                        f"{pos_icons[pos]} **{pos.capitalize()}:** "
                        + " · ".join(labels)
                    )
            else:
                st.info(
                    "ℹ️ No objects detected. "
                    "Try adjusting the threshold or recapturing."
                )

        with tab_visual:
            if screen_reader_mode:
                st.info("🖼️ Annotated image hidden in screen-reader mode.")
            elif st.session_state.last_annotated_frame is not None:
                st.image(
                    st.session_state.last_annotated_frame,
                    channels="BGR",
                    caption="AI Detection Results",
                    use_container_width=True,
                )
            else:
                st.info("Capture an image to see the annotated visual.")

        with tab_details:
            if detections:
                counts = Counter(d["name"] for d in detections)
                st.markdown("**Objects by class:**")
                for cls_name, cnt in sorted(counts.items(), key=lambda x: -x[1]):
                    badge = (
                        " 🚨"
                        if (cls_name in HAZARD_CLASSES and hazard_mode)
                        else ""
                    )
                    st.write(
                        f"- **{cls_name}**{badge}: "
                        f"{cnt} instance{'s' if cnt > 1 else ''}"
                    )

                st.markdown("**All detections (sorted by size):**")
                for i, d in enumerate(
                    sorted(detections, key=lambda d: d["area_ratio"], reverse=True),
                    1,
                ):
                    hazard_tag = (
                        " 🚨 hazard" if (d["is_hazard"] and hazard_mode) else ""
                    )
                    st.write(
                        f"{i}. **{d['name']}**{hazard_tag} — "
                        f"{d['position']}, {d['distance']} "
                        f"(conf: {d['conf'] * 100:.0f}%)"
                    )
            else:
                st.info("No detections to display.")

        # ── Audio playback ────────────────────────────────────────────────────
        st.subheader("Step 4 — Voice output")

        if st.session_state.last_audio is not None:
            st.audio(
                st.session_state.last_audio,
                format="audio/mp3",
                autoplay=autoplay,
            )
            col_replay, _ = st.columns([1, 3])
            with col_replay:
                if st.button("🔁 Replay audio"):
                    st.audio(
                        st.session_state.last_audio,
                        format="audio/mp3",
                        autoplay=True,
                    )
            st.caption(
                "Use the player above to control playback. "
                "Toggle *Autoplay audio* in the sidebar to change default behaviour."
            )
        else:
            st.warning(
                "Audio unavailable. Check your network connection and try again."
            )

        # ── Feedback section ──────────────────────────────────────────────────
        st.divider()
        st.subheader("📝 Submit Feedback")
        st.caption(
            "Help us improve by reporting wrong detections or other issues."
        )

        sheets_ok, sheets_err = check_sheets_config()
        if sheets_err:
            st.warning(
                f"⚠️ Feedback storage is not configured: {sheets_err}  "
                "You can still fill in the form but submission is disabled."
            )

        with st.form("feedback_form", clear_on_submit=True):
            fb_category = st.selectbox("Category *", FEEDBACK_CATEGORIES)
            fb_comment = st.text_area(
                "Comment *",
                placeholder="Describe the problem in detail…",
                help="Required. What went wrong?",
            )
            fb_contact = st.text_input(
                "Email / contact (optional)",
                placeholder="you@example.com",
            )
            fb_include_image = st.checkbox(
                "Include last captured image in feedback",
                value=False,
                help="The image will be stored as compressed data in the feedback sheet.",
            )
            st.caption(
                "🔒 **Privacy:** the image is only uploaded if the checkbox above is checked."
            )

            submitted = st.form_submit_button(
                "Submit Feedback",
                disabled=(not sheets_ok),
            )

        if submitted:
            if not fb_comment.strip():
                st.error("Please enter a comment before submitting.")
            else:
                dets = st.session_state.last_detections or []
                image_b64 = ""
                if fb_include_image and st.session_state.last_image_bytes:
                    image_b64 = encode_image_for_sheet(
                        st.session_state.last_image_bytes
                    )

                row = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "app_version": APP_VERSION,
                    "category": fb_category,
                    "comment": fb_comment.strip(),
                    "user_contact": fb_contact.strip(),
                    "hazard_mode": str(hazard_mode),
                    "conf_threshold": str(conf_threshold),
                    "language_label": lang_label,
                    "language_code": tts_lang,
                    "slow_speech": str(slow_speech),
                    "screen_reader_mode": str(screen_reader_mode),
                    "spoken_sentence": st.session_state.last_sentence or "",
                    "detection_summary": json.dumps(
                        [
                            {
                                "name": d["name"],
                                "conf": round(d["conf"], 3),
                                "position": d["position"],
                                "distance": d["distance"],
                                "is_hazard": d["is_hazard"],
                                "area_ratio": round(d["area_ratio"], 4),
                            }
                            for d in dets
                        ]
                    ),
                    "class_counts": json.dumps(
                        dict(Counter(d["name"] for d in dets))
                    ),
                    "image_b64": image_b64,
                }

                with st.spinner("Submitting feedback…"):
                    success, err = append_feedback_to_sheet(row)

                if success:
                    st.success("✅ Feedback submitted — thank you!")
                else:
                    st.error(f"❌ Submission failed: {err}")


# ══════════════════════════════════════════════════════════════════════════════
# LIVE VIDEO MODE
# ══════════════════════════════════════════════════════════════════════════════

else:  # app_mode == "Live video"
    st.markdown(
        "**Live video mode** — YOLO runs on every few frames and overlays "
        "bounding boxes in real time."
    )

    if not WEBRTC_AVAILABLE:
        st.error(
            "⚠️ `streamlit-webrtc` is not installed. "
            "Install it with:\n\n"
            "```\npip install streamlit-webrtc av\n```\n\n"
            "Then restart the app."
        )
        st.stop()

    webrtc_ctx = webrtc_streamer(
        key="live-yolo",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=YOLOVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if webrtc_ctx.video_processor:
        webrtc_ctx.video_processor.conf_threshold = conf_threshold
        webrtc_ctx.video_processor.hazard_mode = hazard_mode

    st.caption(
        "Adjust *Confidence threshold* and *Safety / hazard mode* in the "
        "sidebar to change detection behaviour."
    )
    st.divider()

    live_text_placeholder = st.empty()
    live_audio_placeholder = st.empty()

    if webrtc_ctx.state.playing:
        last_speech_time = 0.0

        while True:
            if webrtc_ctx.video_processor is None:
                break

            # Sync sidebar controls to processor
            webrtc_ctx.video_processor.conf_threshold = conf_threshold
            webrtc_ctx.video_processor.hazard_mode = hazard_mode

            live_dets = webrtc_ctx.video_processor.get_detections()

            sentence = summarize_detections(live_dets, hazard_mode)
            groups = build_grouped_summary(live_dets, hazard_mode)

            with live_text_placeholder.container():
                if live_dets:
                    hazards = [d for d in live_dets if d["is_hazard"]]
                    if hazards and hazard_mode:
                        st.warning(
                            "⚠️ **Hazards:** "
                            + ", ".join(
                                f"{d['name']} ({d['position']}, {d['distance']})"
                                for d in hazards
                            )
                        )
                    pos_icons = {"left": "⬅️", "center": "⬆️", "right": "➡️"}
                    for pos in ("left", "center", "right"):
                        items = groups[pos]
                        if not items:
                            continue
                        labels = [
                            f"{'🚨 ' if d['is_hazard'] and hazard_mode else ''}"
                            f"**{d['name']}** ({d['distance']})"
                            for d in items
                        ]
                        st.write(
                            f"{pos_icons[pos]} {pos.capitalize()}: "
                            + " · ".join(labels)
                        )
                else:
                    st.info("👁️ Scanning… no objects detected yet.")

            # Throttled speech
            now = time.monotonic()
            if (
                speak_alerts_live
                and live_dets
                and (now - last_speech_time) >= LIVE_SPEECH_INTERVAL
            ):
                try:
                    audio = generate_tts_audio(
                        sentence, lang=tts_lang, slow=slow_speech
                    )
                    live_audio_placeholder.audio(
                        audio, format="audio/mp3", autoplay=True
                    )
                    last_speech_time = now
                except RuntimeError:
                    pass

            time.sleep(1.0)

            if not webrtc_ctx.state.playing:
                break
