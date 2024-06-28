import os
import shutil

# Define the source directory where the original videos are stored
source_dir = '../../data/local_video_action_dataset/final'

# Define the target directory where filtered videos will be saved
target_dir = '../../data/local_video_action_dataset/final_fiveLabels'

# Create target directory if it doesn't exist
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# List of desired labels
desired_labels = [
    'Grazen',
    'Herkauwen(staand)',
    'Liggen(rusten)',
    'Liggen(herkauwen)',
    'Staan'
]

# Function to extract the label from the video filename
def extract_label(filename):
    # Split the filename and return the label part
    parts = filename.split('_')
    if len(parts) > 1:
        return parts[1]
    return None

# Iterate over all files in the source directory
for filename in os.listdir(source_dir):
    if filename.endswith(".avi"):  # Check if the file is a video
        label = extract_label(filename)
        if label in desired_labels:
            # If the label is one of the desired labels, copy the file to the target directory
            shutil.copy(os.path.join(source_dir, filename), os.path.join(target_dir, filename))

print("Filtering complete.")
