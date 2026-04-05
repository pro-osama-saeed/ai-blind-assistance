# AI Blind Assistance

An assistive application that helps blind and low-vision users understand their surroundings using AI—through voice-first interaction, real-time scene understanding, and accessible guidance.

## Overview

**AI Blind Assistance** aims to improve day-to-day independence by providing quick, spoken feedback about the environment. The project focuses on accessibility, low friction UX, and practical features that work in real-world conditions.

## Key Features (Planned / In Progress)

- **Voice-first experience** (hands-free)
  - Speak commands like: *“What’s in front of me?”* or *“Read this text.”*
- **Scene description**
  - Summarizes what the camera sees (objects, people, environment context).
- **Object detection & guidance**
  - Identifies important objects (doors, chairs, stairs, etc.) and can provide directional hints.
- **Text reading (OCR)**
  - Reads printed text from signs, labels, documents.
- **Accessible UI**
  - Works smoothly with screen readers and large text settings.

> If some features are not implemented yet, keep them under “Planned” to set clear expectations.

## Use Cases

- Identifying objects in a room
- Reading signs/labels and short documents
- Getting a quick spoken description of the surroundings
- Assistance during indoor navigation (future scope)

## Tech Stack

- **Frontend:** Streamlit (Python web app framework)
- **AI / Detection:** YOLOv8n via `ultralytics`, OpenCV
- **Speech:** Google Text-to-Speech (`gTTS`)
- **Live video:** `streamlit-webrtc`, `av` (PyAV)
- **Feedback storage:** Google Sheets via `gspread` + `google-auth`

## Getting Started

### Prerequisites
- Python 3.10+
- A webcam or image files for testing
- (Optional) A Google Cloud service account and a Google Sheet for feedback storage

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

The app opens in your browser at `http://localhost:8501`.

---

## Google Sheets Feedback Setup (optional)

The feedback form stores submissions in a Google Sheet. This is **optional** — the app runs fully in *Single capture* mode without it.

### 1. Create a Google Cloud service account

1. Go to [Google Cloud Console](https://console.cloud.google.com/) → **IAM & Admin → Service Accounts**.
2. Create a new service account (name it anything, e.g. `blind-assist-feedback`).
3. Give it no roles at project level (permissions are handled via the sheet).
4. Create a JSON key: **Manage keys → Add key → JSON**. Download the file.

### 2. Enable the Sheets & Drive APIs

In your project, enable:
- **Google Sheets API**
- **Google Drive API**

### 3. Share the Google Sheet with the service account

1. Create a new Google Sheet (or use an existing one).
2. Click **Share** and add the service account email (found in the JSON key file, field `client_email`) with **Editor** access.
3. Copy the Sheet ID from the URL: `https://docs.google.com/spreadsheets/d/<SHEET_ID>/edit`.

### 4. Add Streamlit secrets

Create `.streamlit/secrets.toml` in the project root (this file is git-ignored — **never commit it**):

```toml
google_sheet_id = "YOUR_SHEET_ID_HERE"

[gcp_service_account]
type = "service_account"
project_id = "your-project-id"
private_key_id = "key-id"
private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
client_email = "blind-assist-feedback@your-project.iam.gserviceaccount.com"
client_id = "123456789"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/..."
```

> All values come from the downloaded JSON key file. Copy them exactly.

When secrets are present, the feedback form's **Submit** button becomes active and appends rows to a `Feedback` worksheet (created automatically on first submission).

---

## Live Video Mode

Switch to **Live video** in the sidebar to start continuous detection from your webcam.

- Requires `streamlit-webrtc` and `av` (included in `requirements.txt`).
- YOLO runs every few frames to keep CPU usage manageable.
- Enable **Auto-speak alerts** in *Audio controls* to hear spoken summaries at most every 5 seconds.
- Adjust *Confidence threshold* and *Safety / hazard mode* in real time from the sidebar.

## Project Structure (Optional)

```text
ai-blind-assistance/
  README.md
  (add folders here once they exist)
```

## Accessibility Notes

This project is designed with accessibility as a priority:

- Clear voice prompts and short responses
- Minimal on-screen interaction required
- High-contrast friendly design (if UI exists)
- Compatible with screen readers (TalkBack/VoiceOver) where applicable

## Roadmap

- [ ] Basic camera capture + voice command trigger
- [ ] Scene description model integration
- [ ] OCR reading mode
- [ ] Object detection with spoken results
- [ ] Offline mode (where possible)
- [ ] User testing + feedback iterations

## Contributing

Contributions are welcome.

1. Fork the repo  
2. Create a feature branch: `git checkout -b feature/my-feature`  
3. Commit changes  
4. Open a Pull Request

## License

Add a license (recommended: MIT) and update this section accordingly.

---

## What I need from you to finalize it
Reply with:
1) Is this an **Android app**, **web app**, **Python project**, or something else?  
2) What AI services/models are you using (if any)?  
3) What features are already done vs planned?

And I’ll rewrite the README to match your exact implementation and add correct install/run instructions.
