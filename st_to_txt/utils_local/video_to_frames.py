import cv2
import numpy as np
import os

def preprocess_video(url, start_time_seconds):
    """
    This function opens the video and cuts the right half of each frame starting from 
    the given start time. It then yields the right half of each frame for further processing.
    
    :param url: URL or path to the input video
    :param start_time_seconds: Start time of the video in seconds (when the weather report begins)
    :return: Generator yielding the right half of each frame
    """
    # Open the video
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {url}")
        return None
    
    # Set the video position to the specified start time
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time_seconds * 1000)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Cut the right half of the frame
        height, width, _ = frame.shape
        right_half = frame[:, width // 2:]
        
        # Yield the right half of the frame to be processed further
        yield right_half

    # Release the video capture object after processing
    cap.release()


def save_video_frames(url, start_time_seconds, output_folder, fps=25, frame_size=(210, 260)):
    """
    This function extracts frames from the video, cuts the right half, resizes them, and saves them 
    to the specified folder at the specified frame rate and size.
    
    :param url: URL or path to the input video
    :param start_time_seconds: Start time of the video in seconds (when the weather report begins)
    :param output_folder: Folder where the frames should be saved
    :param fps: Frames per second for extraction (default is 25)
    :param frame_size: Tuple representing the size of the saved frames (default is (210, 260))
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frame_count = 0
    # Process video using preprocess_video to get right half of each frame
    for right_half in preprocess_video(url, start_time_seconds):
        # Resize the right half to the desired size (210x260)
        frame_resized = cv2.resize(right_half, frame_size)
        
        # Save frame as an image file
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_filename, frame_resized)
        
        frame_count += 1
        
        # Wait for the specified amount of time to maintain fps
        cv2.waitKey(int(1000 / fps))

    print(f"Frames saved to {output_folder}")


# Example URL (change it to your actual video file or stream URL)
video_url = "https://wdrvod-rwrtr.akamaized.net/i/,/medp/ondemand/weltweit/fsk0/323/3235010/,3235010_60750797,3235010_60750798,3235010_60750796,3235010_60750799,3235010_60750795,.mp4.csmil/index-f5-v1-a1.m3u8"

# Example start time (13:54 = 834 seconds)
start_time = 13 * 60 + 54  # 834 seconds

# Save frames to a folder
output_folder = "st_to_txt/data/extracted_frames"
save_video_frames(video_url, start_time, output_folder)