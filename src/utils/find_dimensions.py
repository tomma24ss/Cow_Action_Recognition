import cv2

# Path to the video file
video_path = '../../data/local_video_action_dataset/final_fiveLabels/v_Staan_cltczm2lf1p700701tmhj26ya_clxneboar06h0193f4smv7dio.avi'

# Open the video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
else:
    # Get the dimensions of the video frames
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Print the dimensions
    print(f"WIDTH = {width}")
    print(f"HEIGHT = {height}")

    # Release the video capture object
    cap.release()
