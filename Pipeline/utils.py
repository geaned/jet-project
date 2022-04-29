import gdown
import os

def download_if_file_not_present(google_id: str, file_path: str):
    if os.path.exists(file_path):
        print(f'Found {file_path}!')
        return
    
    print(f'Could not find {file_path}! Downloading...')
    gdown.download('https://drive.google.com/uc?id='+google_id, file_path)
