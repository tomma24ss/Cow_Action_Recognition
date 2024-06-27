import json
import cv2
import requests
import logging
import os
import shutil
import traceback
from uuid import uuid4
from collections import defaultdict
from dotenv import load_dotenv
import sys

# Configure logging
log_file = 'video_action_dataset_manager.log'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(file_handler)

class DiskSpaceError(Exception):
    pass

class VideoActionDatasetManager:
    def __init__(self):
        self.history_file = 'processing_history.json'
        self.processed_videos = self.load_processing_history()
        self.min_disk_space = 100 * 1024 * 1024  # Minimum 100MB free space required

    def download_video(self, url, destination_path):
        """Download video from URL to the specified destination path."""
        try:
            logging.info(f'Downloading video from {url}...')
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Ensure we catch any HTTP errors

            content_type = response.headers.get('Content-Type')
            if 'video' not in content_type:
                logging.error(f'URL does not point to a video file. Content-Type: {content_type}')
                return None

            content_length = int(response.headers.get('Content-Length', 0))
            
            with open(destination_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1048576):  # Use 1MB chunks
                    if chunk:
                        f.write(chunk)

            downloaded_size = os.path.getsize(destination_path)
            if downloaded_size != content_length:
                logging.error(f'Downloaded file size mismatch: expected {content_length}, got {downloaded_size}')
                return None
            
            logging.info(f'Video downloaded successfully to {destination_path}.')
            return destination_path
        except requests.exceptions.RequestException as e:
            logging.error(f'Failed to download video from {url}. Exception: {e}')
            return None
        except Exception as e:
            logging.error(f'Unexpected error during download: {e}')
            return None

    def get_or_download_video(self, url, video_id, folder='videos'):
        """Get video from folder or download it if it doesn't exist."""
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        video_path = os.path.join(folder, f'{video_id}.mp4')
        
        if os.path.exists(video_path):
            logging.info(f'Video {video_id} already exists. Using the existing video.')
            return video_path
        else:
            return self.download_video(url, video_path)

    def load_processing_history(self):
        """Load the processing history from a file."""
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return {}

    def save_processing_history(self):
        """Save the processing history to a file."""
        with open(self.history_file, 'w') as f:
            json.dump(self.processed_videos, f, indent=4)

    def check_disk_space(self, path):
        """Check if there is enough disk space available."""
        total, used, free = shutil.disk_usage(path)
        if free < self.min_disk_space:
            raise DiskSpaceError("Not enough disk space available.")

    def process_annotations(self, ndjson_file_path, local_dataset_folder):
        """Process annotations from an NDJSON file and save them locally."""
        processing_folder = os.path.join(local_dataset_folder, 'processing')
        final_folder = os.path.join(local_dataset_folder, 'final')
        
        if not os.path.exists(processing_folder):
            os.makedirs(processing_folder)
        if not os.path.exists(final_folder):
            os.makedirs(final_folder)
        
        try:
            with open(ndjson_file_path, 'r') as f:
                data = [json.loads(line) for line in f]
        except Exception as e:
            logging.error(f"Failed to load NDJSON file: {e}")
            raise

        try:
            for index, item in enumerate(data):
                data_row_id = item['data_row']['id']

                # Skip already processed videos
                if data_row_id in self.processed_videos:
                    logging.info(f"Video {data_row_id} has already been processed. Skipping.")
                    continue

                try:
                    video_url = item['data_row']['row_data']
                    labels = item['projects']['clxk41sdz01gl07u0025q4ngv']['labels']

                    if not labels or len(labels) == 0:
                        logging.warning(f"No labels found for video {data_row_id}. Skipping.")
                        continue

                    video_path = self.get_or_download_video(video_url, data_row_id)
                    if video_path is None:
                        continue
                    
                    cap = cv2.VideoCapture(video_path)
                    if not cap.isOpened():
                        logging.error(f'Failed to open video stream from {video_url}')
                        continue
                    
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    logging.info(f'Reading video from {video_url} with total frames {total_frames}...')

                    writers = defaultdict(lambda: None)
                    current_actions = {}
                    action_counts = defaultdict(int)

                    try:
                        for label in labels:
                            annotation = label['annotations']
                            if not annotation or 'frames' not in annotation:
                                logging.warning(f"No frames found in annotations for video {data_row_id}. Skipping label.")
                                continue

                            for frame_number, frame_data in annotation['frames'].items():
                                frame_number = int(frame_number)
                                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                                ret, frame = cap.read()
                                if not ret:
                                    logging.warning(f'Frame {frame_number} could not be read. Ending processing.')
                                    break

                                logging.info(f'Processing frame {frame_number} for video {data_row_id}... (frame {frame_number}/{total_frames}, video {index + 1}/{len(data)})')

                                for obj_id, obj in frame_data['objects'].items():
                                    bbox = obj['bounding_box']
                                    action = obj['name'].replace(" ", "")

                                    left = int(bbox['left'])
                                    top = int(bbox['top'])
                                    width = int(bbox['width'])
                                    height = int(bbox['height'])

                                    padding = 10  # Padding value
                                    x1 = max(left - padding, 0)
                                    y1 = max(top - padding, 0)
                                    x2 = min(left + width + padding, frame.shape[1])
                                    y2 = min(top + height + padding, frame.shape[0])

                                    cropped_frame = frame[y1:y2, x1:x2]

                                    # Check if the action has changed for the object
                                    if obj_id not in current_actions or current_actions[obj_id] != action:
                                        # Close the existing writer if there is one
                                        if writers[obj_id] is not None:
                                            writers[obj_id]['writer'].release()
                                            shutil.move(writers[obj_id]['path'], final_folder)
                                            logging.info(f'Moved {writers[obj_id]["path"]} from processing to final dataset folder')

                                        # Increment the action count for the object
                                        action_counts[obj_id] += 1
                                        
                                        # Create a new writer for the new action
                                        video_filename = f'v_{action}_{data_row_id}_{obj_id}.avi'
                                        cropped_video_path = os.path.join(processing_folder, video_filename)
                                        writer = cv2.VideoWriter(
                                            cropped_video_path, cv2.VideoWriter_fourcc(*'XVID'), 30, 
                                            (cropped_frame.shape[1], cropped_frame.shape[0])
                                        )
                                        writers[obj_id] = {
                                            'path': cropped_video_path,
                                            'writer': writer,
                                            'action': action,
                                        }
                                        current_actions[obj_id] = action
                                        logging.info(f'Initialized writer for {obj_id} with filename {video_filename}')

                                    # Check disk space before writing the frame
                                    self.check_disk_space(processing_folder)
                                    
                                    # Write the frame to the current writer
                                    writers[obj_id]['writer'].write(cropped_frame)
                                    logging.info(f'Written frame {frame_number} for object {obj_id} to {writers[obj_id]["path"]}')

                    finally:
                        cap.release()

                        # Release all writers and move videos to final folder
                        for obj_id, writer_info in writers.items():
                            if writer_info is not None:
                                writer_info['writer'].release()
                                shutil.move(writer_info['path'], final_folder)
                                logging.info(f'Moved {writer_info["path"]} from processing to final dataset folder')

                        self.processed_videos[data_row_id] = True
                        self.save_processing_history()

                except DiskSpaceError as e:
                    logging.error(f"Disk space error: {e}")
                    raise
                except KeyboardInterrupt:
                    logging.info("Process interrupted by user. Exiting gracefully...")
                    sys.exit(0)
                except Exception as e:
                    logging.error(f"Error processing item {str(item)[:50]}: {e}\n{traceback.format_exc()}")
                    raise
                finally:
                    # Ensure the video file is deleted after processing
                    if os.path.exists(video_path):
                        os.remove(video_path)
                        logging.info(f'Deleted video file {video_path} to free up space.')

        except KeyboardInterrupt:
            logging.info("Process interrupted by user. Exiting gracefully...")
            sys.exit(0)

# Usage Example
load_dotenv()
manager = VideoActionDatasetManager()

# Process annotations from NDJSON file
ndjson_file_path = "../data/labelbox_exports/video_dataset.ndjson"
local_dataset_folder = '../data/local_video_action_dataset'
manager.process_annotations(ndjson_file_path, local_dataset_folder)
