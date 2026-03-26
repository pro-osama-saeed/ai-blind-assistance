import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import collections
from gtts import gTTS
import os

# Set up the webpage structure
st.title("Blind Assistance: Environment Scanner")
st.write("Click the button below to scan your surroundings. The AI will detect objects and speak the results.")

# Load the AI Model 
# (@st.cache_resource ensures the model only loads once, keeping the website fast!)
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

# This single line replaces all of the Colab JavaScript!
camera_image = st.camera_input("Capture and Scan Environment")

if camera_image is not None:
    st.write("Scanning Image with YOLOv8...")

    # Convert the web camera image into a format OpenCV and YOLO can read
    bytes_data = camera_image.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # Pass the image to YOLOv8
    results = model(cv2_img, verbose=False)

    # Extract the object names
    detected_names = []
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        detected_names.append(class_name)

    # Draw the bounding boxes and display the final image on the website
    annotated_frame = results[0].plot()
    
    # Streamlit expects RGB color formatting, not OpenCV's default BGR
    st.image(annotated_frame, channels="BGR", caption="AI Detection Results")

    # 4. Counting and Audio Output
    if detected_names:
        item_counts = dict(collections.Counter(detected_names))
        
        speech_parts = [f"{count} {item}{'s' if count > 1 else ''}" for item, count in item_counts.items()]
        spoken_sentence = "I detect " + ", ".join(speech_parts)
        
        st.success(f"**Voice Output:** {spoken_sentence}")
        
        # Generate the audio file
        tts = gTTS(text=spoken_sentence, lang='en', slow=False)
        tts.save("feedback.mp3")
        
        # Play the audio directly in the web browser
        st.audio("feedback.mp3", format="audio/mp3", autoplay=True)
        
    else:
        st.warning("**Voice Output:** No objects detected in the current view.")
        tts = gTTS(text="No objects detected in the current view.", lang='en', slow=False)
        tts.save("feedback.mp3")
        st.audio("feedback.mp3", format="audio/mp3", autoplay=True)
