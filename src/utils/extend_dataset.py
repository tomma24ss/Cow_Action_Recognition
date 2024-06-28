from moviepy.editor import VideoFileClip
import os

# Define the directory containing the original videos
source_dir = '../../data/local_video_action_dataset/final_fiveLabels'

# Define the target directory where filtered videos will be saved
target_dir = '../../data/local_video_action_dataset/final_fiveLabels_splitted'

# Create the target directory if it doesn't exist
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# Duration of each split video in seconds
split_duration = 3

# Function to split the video
def split_video(video_path, filename):
    # Load the video file
    clip = VideoFileClip(video_path)
    
    # Get the total duration of the video in seconds
    total_duration = int(clip.duration)
    
    if total_duration > 3:
        # Calculate the number of splits
        num_splits = total_duration // split_duration
        
        # Generate splits
        for i in range(num_splits):
            # Calculate start and end times
            start_time = i * split_duration
            end_time = start_time + split_duration
            
            # Create a subclip
            subclip = clip.subclip(start_time, end_time)
            
            # Construct new filename
            new_filename = f"{filename}-{i + 1}.avi"
            new_file_path = os.path.join(target_dir, new_filename)
            
            # Write the subclip to a file
            subclip.write_videofile(new_file_path, codec='libx264')
            
            # Print progress
            print(f"Created: {new_filename}")
    else:
        # For videos shorter than 3 seconds, move them without splitting
        new_file_path = os.path.join(target_dir, filename)
        clip.write_videofile(new_file_path, codec='libx264')
        print(f"Moved: {filename}")

    # Close the video file to free resources
    clip.close()

# Process all video files in the directory
for filename in os.listdir(source_dir):
    if filename.endswith(".avi"):  # Check if the file is a video
        video_path = os.path.join(source_dir, filename)
        split_video(video_path, filename)

print("Video splitting complete.")
