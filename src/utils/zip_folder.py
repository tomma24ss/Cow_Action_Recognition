import os
import zipfile

def zip_folder(folder_path, output_path):
    # Create a ZipFile object
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # The root directory within the ZIP file
        root_len = len(os.path.abspath(folder_path))
        
        # Walk the directory
        for root, dirs, files in os.walk(folder_path):
            archive_root = os.path.abspath(root)[root_len:]
            
            # Add each file in the folder
            for file in files:
                fullpath = os.path.join(root, file)
                archive_name = os.path.join(archive_root, file)
                zipf.write(fullpath, archive_name)
                
    print(f"Folder '{folder_path}' has been zipped into '{output_path}'")

# Example usage
folder_to_zip = "../../data/local_video_action_dataset/final"  # Change this to the path of the folder you want to zip
output_zip_file = "../../data/local_video_action_dataset/CowActionNet.zip"  # Change this to your desired zip file path
zip_folder(folder_to_zip, output_zip_file)
