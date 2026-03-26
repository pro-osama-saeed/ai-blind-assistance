# AI Blind Assistance

A voice-first web application that helps blind and low-vision users understand their surroundings using real-time AI — through object detection, text reading (OCR), and natural scene descriptions.

## Overview

**AI Blind Assistance** provides instant spoken feedback about the environment. Point your camera, choose a mode, and the app speaks what it sees. Designed for accessibility-first usage with minimal on-screen interaction.

## Features

| Feature | Status | Description |
|---------|--------|-------------|
| **Object Detection** | Done | Detects objects and reports their position (left / ahead / right) with confidence filtering |
| **Spatial Position Hints** | Done | Every detected object is tagged as left, center, or right |
| **Text Reading (OCR)** | Done | Reads printed text from signs, labels, and documents using EasyOCR |
| **Scene Description** | Done | Generates a natural spoken description (template-based or GPT-4o mini) |
| **Voice Command Input** | Done | Web Speech API button — say "scan", "read text", or "describe" |
| **High-Contrast Mode** | Done | Black background / yellow text toggle for low-vision users |
| **Configurable Model** | Done | Choose Nano (fastest), Small (balanced), or Medium (most accurate) YOLO model |
| **Confidence Threshold** | Done | Sidebar slider to filter out low-confidence detections |
| **Multi-language TTS** | Done | English, Arabic, French, Spanish, German, Urdu |
| **Offline Mode** | Planned | Offline TTS + bundled model |
| **Mobile App** | Planned | iOS / Android wrapper |

## Tech Stack

- **UI:** Streamlit (Python web framework)
- **Object Detection:** YOLOv8 via Ultralytics (Nano / Small / Medium)
- **OCR:** EasyOCR (PyTorch-based, no system dependencies)
- **Scene Description:** Template-based + optional OpenAI GPT-4o mini
- **Speech Output:** Google Text-to-Speech (gTTS)
- **Voice Input:** Web Speech API (browser-native, Chrome/Edge)
- **Image Processing:** OpenCV

## Getting Started

### Prerequisites

- Python 3.10+
- A webcam or mobile camera
- Internet connection (for gTTS; optional for GPT scene descriptions)

### Installation

```bash
git clone https://github.com/pro-osama-saeed/ai-blind-assistance.git
cd ai-blind-assistance
pip install -r requirements.txt
```

### Run

```bash
streamlit run app.py
```

Open http://localhost:8501 in Chrome or Edge (required for voice commands).

On first run, the YOLO model weights are downloaded automatically (~6 MB for Nano, ~22 MB for Small).

## Project Structure

```
ai-blind-assistance/
├── app.py              # Main application
├── requirements.txt    # Python dependencies
└── README.md
```

## Usage

1. **Choose a mode** — Scan Objects, Read Text, or Describe Scene
2. (Optional) **Say a voice command** — "scan", "read text", or "describe"
3. **Take a photo** using the camera widget
4. **Listen** — the app speaks the result automatically

### Sidebar Options

| Setting | Description |
|---------|-------------|
| High-Contrast Mode | Black/yellow UI for low-vision users |
| Detection Model | Nano (fast) / Small (balanced) / Medium (accurate) |
| Confidence Threshold | Filter out uncertain detections (default 50%) |
| Voice Language | Language for spoken output |
| OpenAI API Key | Optional — enables GPT-4o mini scene descriptions |

## Accessibility Notes

- Voice-first design: all feedback is spoken automatically
- Minimal on-screen interaction required
- High-contrast mode for low-vision users
- Large buttons with clear labels
- Compatible with browser screen readers (VoiceOver / NVDA)

## Roadmap

- [x] Camera capture + object detection
- [x] Spatial position hints (left / ahead / right)
- [x] Confidence threshold filtering
- [x] Scene description (template + LLM)
- [x] OCR text reading
- [x] Voice command input
- [x] High-contrast UI mode
- [x] Multi-language voice output
- [ ] Offline mode (pyttsx3 + bundled model)
- [ ] Hazard/obstacle alert sounds
- [ ] Mobile app (React Native / Flutter)

## Contributing

Contributions are welcome.

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit changes
4. Open a Pull Request

## License

MIT
