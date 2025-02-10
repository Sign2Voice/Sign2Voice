import os
import cv2
from datetime import datetime

def save_frames_from_stream(frame, output_folder, source_fps=30, save_fps=25, frame_size=(210, 260), frame_count=0):
    """
    Saves frames from a stream based on the desired save rate (save_fps).
    :param frame: Single frame from the stream
    :param output_folder: Folder to save the frames
    :param source_fps: FPS of the source
    :param save_fps: Number of frames per second to save
    :param frame_size: Tuple (width, height) for the frame size
    :param frame_count: Current frame counter
    :return: Next frame counter
    """
    
    # Calculate the interval between saved frames
    frame_interval = source_fps // save_fps

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Check if the current frame should be saved
    if frame_count % frame_interval == 0:
        # Adjust frame size
        frame_resized = cv2.resize(frame, frame_size)

        # Save frame
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        frame_filename = os.path.join(output_folder, f"frame_{timestamp}_{frame_count:04d}.png")
        cv2.imwrite(frame_filename, frame_resized)

    return frame_count + 1