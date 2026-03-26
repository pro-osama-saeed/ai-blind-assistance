import streamlit as st
import streamlit.components.v1 as components
import cv2
import numpy as np
from ultralytics import YOLO
import collections
from gtts import gTTS

# ─── Page configuration ────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Blind Assistance",
    page_icon="👁️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ─── Session state defaults ────────────────────────────────────────────
if "mode" not in st.session_state:
    st.session_state.mode = "scan"
if "high_contrast" not in st.session_state:
    st.session_state.high_contrast = False

# ─── Dynamic CSS (normal & high-contrast) ─────────────────────────────
def apply_styles(high_contrast: bool) -> None:
    if high_contrast:
        css = """
        <style>
        .stApp, [data-testid="stAppViewContainer"] {
            background-color: #000000 !important;
            color: #FFD700 !important;
        }
        h1, h2, h3, p, label, span,
        .stMarkdown, .stCaption, [data-testid="stText"] {
            color: #FFD700 !important;
        }
        .stButton > button {
            background-color: #FFD700 !important;
            color: #000000 !important;
            font-size: 1.3rem !important;
            padding: 0.75rem 2rem !important;
            border-radius: 8px !important;
            font-weight: bold !important;
            border: 2px solid #FFD700 !important;
        }
        .stButton > button:hover { opacity: 0.85; }
        [data-testid="stSidebar"] {
            background-color: #111111 !important;
        }
        [data-testid="stSidebar"] * { color: #FFD700 !important; }
        </style>
        """
    else:
        css = """
        <style>
        .stButton > button {
            font-size: 1.05rem !important;
            padding: 0.55rem 1.4rem !important;
            border-radius: 8px !important;
        }
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)


# ─── Model loading (cached per size selection) ─────────────────────────
MODEL_OPTIONS = {
    "Nano — fastest":          "yolov8n.pt",
    "Small — balanced ✓":      "yolov8s.pt",
    "Medium — most accurate":  "yolov8m.pt",
}

@st.cache_resource
def load_model(model_key: str) -> YOLO:
    return YOLO(MODEL_OPTIONS[model_key])


# ─── OCR reader (cached, loaded on demand) ────────────────────────────
@st.cache_resource
def load_ocr():
    import easyocr  # heavy import; only when OCR mode is selected
    return easyocr.Reader(["en"], gpu=False)


# ─── Helper: spatial position from bounding box ───────────────────────
def get_position(box, img_width: int) -> str:
    x1, _y1, x2, _y2 = box.xyxy[0].tolist()
    cx = (x1 + x2) / 2
    third = img_width / 3
    if cx < third:
        return "left"
    if cx < 2 * third:
        return "center"
    return "right"


# ─── Helper: pluralisation ────────────────────────────────────────────
# Only YOLO/COCO classes with genuinely irregular plurals are listed here;
# regular suffix rules (e.g. "bus" → "buses") are handled by the function.
_IRREGULAR = {
    "person": "people",
    "knife":  "knives",
    "mouse":  "mice",    # computer mouse (COCO class)
}

def pluralize(word: str, count: int) -> str:
    if count == 1:
        return word
    if word in _IRREGULAR:
        return _IRREGULAR[word]
    if word.endswith(("s", "sh", "ch", "x", "z")):
        return word + "es"
    return word + "s"


# ─── Helper: text-to-speech ───────────────────────────────────────────
def speak(text: str, lang: str = "en") -> None:
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save("feedback.mp3")
    st.audio("feedback.mp3", format="audio/mp3", autoplay=True)


# ─── Helper: scene description ────────────────────────────────────────
def generate_scene_description(
    detections: list[tuple[str, str]], llm_key: str = ""
) -> str:
    """Build a spoken scene description from (name, position) pairs.

    Falls back to a template-based description when no LLM key is provided.
    """
    if not detections:
        return "The scene appears empty — no recognisable objects were found."

    by_pos: dict[str, list[str]] = {"left": [], "center": [], "right": []}
    for name, pos in detections:
        by_pos[pos].append(name)

    # Template-based description
    parts: list[str] = []
    for pos, label in (("center", "directly ahead"), ("left", "on your left"), ("right", "on your right")):
        if by_pos[pos]:
            parts.append(", ".join(by_pos[pos]) + f" {label}")
    template = "In the scene: " + "; ".join(parts) + "."

    if not llm_key:
        return template

    # Optional LLM enhancement via OpenAI
    try:
        from openai import OpenAI  # only imported when key is provided
        client = OpenAI(api_key=llm_key)
        obj_summary = ", ".join(f"{n} ({p})" for n, p in detections)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": (
                    "You are an AI assistant helping a blind person understand their surroundings. "
                    "Based on these detected objects and positions, give a concise 1-2 sentence spoken description: "
                    f"{obj_summary}"
                ),
            }],
            max_tokens=120,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return template  # graceful fallback


# ─── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    st.session_state.high_contrast = st.toggle(
        "🌓 High-Contrast Mode",
        value=st.session_state.high_contrast,
    )

    model_key = st.selectbox(
        "🤖 Detection Model",
        list(MODEL_OPTIONS.keys()),
        index=1,
        help="Larger models are more accurate but take longer to process.",
    )

    conf_threshold = st.slider(
        "🎯 Confidence Threshold",
        min_value=10, max_value=90, value=50, step=5,
        help="Detections below this confidence % are filtered out.",
    )

    language = st.selectbox(
        "🌐 Voice Language",
        ["en", "ar", "fr", "es", "de", "ur"],
        format_func=lambda x: {
            "en": "English", "ar": "Arabic", "fr": "French",
            "es": "Spanish", "de": "German", "ur": "Urdu",
        }[x],
    )

    st.divider()
    st.subheader("🧠 AI Scene Description")
    llm_key = st.text_input(
        "OpenAI API Key (optional)",
        type="password",
        help="Supply a key for richer GPT-powered scene descriptions. Leave blank for template-based descriptions.",
    )

# ─── Apply styles & load model ────────────────────────────────────────
apply_styles(st.session_state.high_contrast)
model = load_model(model_key)

# ─── Page header ──────────────────────────────────────────────────────
st.title("👁️ AI Blind Assistance")
st.caption("Voice-first environmental scanner powered by AI")

# ─── Mode selector ────────────────────────────────────────────────────
st.subheader("Select Mode")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🔍 Scan Objects", use_container_width=True,
                 type="primary" if st.session_state.mode == "scan" else "secondary"):
        st.session_state.mode = "scan"
        st.rerun()
with col2:
    if st.button("📖 Read Text", use_container_width=True,
                 type="primary" if st.session_state.mode == "ocr" else "secondary"):
        st.session_state.mode = "ocr"
        st.rerun()
with col3:
    if st.button("🌍 Describe Scene", use_container_width=True,
                 type="primary" if st.session_state.mode == "describe" else "secondary"):
        st.session_state.mode = "describe"
        st.rerun()

_MODE_INFO = {
    "scan":     "🔍 **Scan Objects** — detects objects and reports their positions (left / ahead / right).",
    "ocr":      "📖 **Read Text** — extracts and reads printed text from signs, labels, and documents.",
    "describe": "🌍 **Describe Scene** — generates a natural spoken description of the whole scene.",
}
st.info(_MODE_INFO[st.session_state.mode])

# ─── Voice command component (Web Speech API) ─────────────────────────
_VOICE_HTML = """
<div style="margin:6px 0 2px 0;">
  <button id="voiceBtn" onclick="startListening()"
    style="font-size:0.95rem;padding:7px 15px;cursor:pointer;
           border-radius:6px;border:2px solid #4CAF50;
           background:#4CAF50;color:#fff;font-weight:bold;">
    🎤 Voice Command
  </button>
  <span id="voiceStatus"
    style="margin-left:10px;font-style:italic;color:#888;font-size:0.9rem;">
    Say "scan", "read text", or "describe"
  </span>
  <script>
  (function() {
    var SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
    var btn = document.getElementById('voiceBtn');
    var status = document.getElementById('voiceStatus');
    var modeMap = {
      'scan': 'Scan Objects',
      'read': 'Read Text', 'read text': 'Read Text',
      'describe': 'Describe Scene', 'describe scene': 'Describe Scene'
    };
    function startListening() {
      if (!SpeechRec) {
        status.textContent = '(Speech recognition not supported — use Chrome/Edge)';
        return;
      }
      var rec = new SpeechRec();
      rec.lang = 'en-US';
      rec.interimResults = false;
      btn.style.background = '#FF5722';
      status.textContent = 'Listening…';
      rec.onresult = function(e) {
        var t = e.results[0][0].transcript.toLowerCase().trim();
        var matched = modeMap[t];
        status.textContent = matched
          ? '✅ Heard: "' + t + '" → click the "' + matched + '" button above, then take a photo.'
          : '❓ Heard: "' + t + '" — say "scan", "read text", or "describe".';
        btn.style.background = '#4CAF50';
      };
      rec.onerror = function() {
        status.textContent = 'Could not hear — please try again.';
        btn.style.background = '#4CAF50';
      };
      rec.onend = function() { btn.style.background = '#4CAF50'; };
      rec.start();
    }
    window.startListening = startListening;
  })();
  </script>
</div>
"""
components.html(_VOICE_HTML, height=55)

# ─── Camera input ─────────────────────────────────────────────────────
st.subheader("📸 Camera")
camera_image = st.camera_input("Point your camera and click to capture")

# ─── Processing ───────────────────────────────────────────────────────
if camera_image is not None:
    bytes_data = camera_image.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    img_height, img_width = cv2_img.shape[:2]

    with st.spinner("Processing…"):

        # ── SCAN mode ─────────────────────────────────────────────────
        if st.session_state.mode == "scan":
            results = model(cv2_img, verbose=False, conf=conf_threshold / 100)

            detections: list[tuple[str, str, float]] = []
            for box in results[0].boxes:
                class_id  = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                position   = get_position(box, img_width)
                detections.append((class_name, position, confidence))

            annotated = results[0].plot()
            st.image(annotated, channels="BGR", caption="Detection Results",
                     use_container_width=True)

            if detections:
                pos_groups: dict[str, list[str]] = {}
                for name, pos, _conf in detections:
                    pos_groups.setdefault(pos, []).append(name)

                parts: list[str] = []
                for pos, loc in (("center", "ahead"), ("left", "on your left"), ("right", "on your right")):
                    if pos in pos_groups:
                        counts = dict(collections.Counter(pos_groups[pos]))
                        obj_str = ", ".join(
                            f"{c} {pluralize(item, c)}" for item, c in counts.items()
                        )
                        parts.append(f"{obj_str} {loc}")

                spoken = "I detect " + "; ".join(parts) + "."
                st.success(f"**Voice Output:** {spoken}")

                with st.expander("📊 Detection Details"):
                    for name, pos, conf in sorted(detections, key=lambda x: -x[2]):
                        bar = "█" * round(conf * 20)
                        st.write(f"**{name}** ({pos}) — {conf:.0%} confidence  {bar}")

                speak(spoken, lang=language)
            else:
                msg = (
                    "No objects detected above the confidence threshold. "
                    "Try lowering the threshold or improving lighting."
                )
                st.warning(f"**Voice Output:** {msg}")
                speak(msg, lang=language)

        # ── OCR mode ──────────────────────────────────────────────────
        elif st.session_state.mode == "ocr":
            st.image(cv2_img, channels="BGR", caption="Image to Read",
                     use_container_width=True)
            try:
                reader = load_ocr()
                rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
                ocr_results = reader.readtext(rgb_img)

                texts = [text for _bbox, text, conf in ocr_results if conf > conf_threshold / 100]
                if texts:
                    full_text = " ".join(texts)
                    st.success(f"**Text Found:** {full_text}")
                    speak(f"I can read: {full_text}", lang=language)
                else:
                    msg = "No readable text found in the image."
                    st.warning(msg)
                    speak(msg, lang=language)
            except ImportError:
                st.error("EasyOCR is not installed. Run: `pip install easyocr`")

        # ── DESCRIBE mode ─────────────────────────────────────────────
        elif st.session_state.mode == "describe":
            results = model(cv2_img, verbose=False, conf=conf_threshold / 100)

            detections_pos: list[tuple[str, str]] = []
            for box in results[0].boxes:
                class_id   = int(box.cls[0])
                class_name = model.names[class_id]
                position   = get_position(box, img_width)
                detections_pos.append((class_name, position))

            annotated = results[0].plot()
            st.image(annotated, channels="BGR", caption="Scene",
                     use_container_width=True)

            description = generate_scene_description(detections_pos, llm_key=llm_key)
            st.info(f"**Scene Description:** {description}")
            if not llm_key:
                st.caption(
                    "💡 Add an OpenAI API key in the sidebar for richer AI-powered descriptions."
                )
            speak(description, lang=language)

# ─── Footer ───────────────────────────────────────────────────────────
st.divider()
st.caption("🔊 Audio plays automatically. Adjust settings in the sidebar.")
