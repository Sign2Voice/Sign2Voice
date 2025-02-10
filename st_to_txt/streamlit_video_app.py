import streamlit as st
import cv2
import numpy as np
import os
import tempfile
import sys

# 🛠 Add root directories to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath("st_to_txt"))

from st_to_txt.process_frames import process_frames  # Function for generating glosses
from PIL import Image

# 📌 Fixed Video URL (must not be changed)
VIDEO_URL = "https://wdrvod-rwrtr.akamaized.net/i/,/medp/ondemand/weltweit/fsk0/323/3235010/,3235010_60750797,3235010_60750798,3235010_60750796,3235010_60750799,3235010_60750795,.mp4.csmil/index-f5-v1-a1.m3u8"

# 📌 Define paths
UPLOAD_FOLDER = "st_to_txt/data/extracted_frames"
PHOENIX_MODEL_PATH = "CorrNet/pretrained_model/dev_18.90_PHOENIX14.pt"

# Ensure the folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def preprocess_video(url, start_time_seconds):
    """
    Extracts the right half of each frame from the video starting from `start_time_seconds`.
    """
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        st.error(f"Error: Unable to open video file {url}")
        return None

    cap.set(cv2.CAP_PROP_POS_MSEC, start_time_seconds * 1000)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        height, width, _ = frame.shape
        right_half = frame[:, width // 2:]  # Extract only the right half
        yield right_half

    cap.release()

def save_video_frames(url, start_time_seconds, output_folder, fps=25, frame_size=(210, 260), max_frames=250):
    """
    Extracts up to `max_frames` frames from the video, resizes them, and saves them.
    """
    frame_count = 0
    for right_half in preprocess_video(url, start_time_seconds):
        if frame_count >= max_frames:
            break  # Stop after extracting 250 frames

        frame_resized = cv2.resize(right_half, frame_size)
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_filename, frame_resized)
        frame_count += 1
        cv2.waitKey(int(1000 / fps))

    return frame_count


# 🎬 **Streamlit App UI**
st.title("📺 German Sign Language Gloss Recognition")

# **Display the fixed video URL**
st.info(f"🔗 **Processing Video from:** {VIDEO_URL}")

# Start time for the weather report (834 seconds)
start_time = 834  

# **Start processing**
if st.button("Extract Frames & Process Glosses"):
    st.write("🔄 Extracting frames from video...")
    num_frames = save_video_frames(VIDEO_URL, start_time, UPLOAD_FOLDER)

    st.success(f"✅ {num_frames} frames extracted and saved.")

    # 🖼 **Preview of extracted frames**
    st.subheader("📸 Frame Preview")
    frame_files = sorted(os.listdir(UPLOAD_FOLDER))[:5]  # Show up to 5 images
    if frame_files:
        images = [Image.open(os.path.join(UPLOAD_FOLDER, img)) for img in frame_files]
        st.image(images, caption=frame_files, width=200)
    else:
        st.warning("No frames extracted!")

    # 🧠 **Generate glosses for the extracted frames**
    st.subheader("🔠 Generating Glosses")
    glosses = process_frames(PHOENIX_MODEL_PATH, UPLOAD_FOLDER, language="phoenix")

    if glosses:
        st.write("📝 **Predicted Glosses:**", " ".join(glosses))
    else:
        st.warning("No glosses generated.")

# 🗑 **Delete frames after processing (optional)**
if st.button("Delete Frames"):
    for file in os.listdir(UPLOAD_FOLDER):
        os.remove(os.path.join(UPLOAD_FOLDER, file))
    st.success("🚮 Frames deleted successfully.")

