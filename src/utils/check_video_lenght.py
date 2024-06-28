import cv2
import os

# Path to the directory containing the videos
directory_path = '../../data/local_video_action_dataset/final_fiveLabels/'

# Function to check the duration of the video
def is_video_longer_than_3_seconds(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return False
    
    # Get the frame rate and the number of frames
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate the duration in seconds
    duration = frame_count / fps if fps > 0 else 0
    
    # Release the video capture object
    cap.release()
    
    return duration < 2

# List all files and print the names of the videos longer than 3 seconds
for filename in os.listdir(directory_path):
    if filename.endswith(".avi"):  # Check if the file is a video file
        video_path = os.path.join(directory_path, filename)
        if is_video_longer_than_3_seconds(video_path):
            print(filename)

