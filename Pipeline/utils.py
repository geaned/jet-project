import gdown
import os

def download_if_file_not_present(google_id: str, file_path: str):
    if os.path.exists(file_path):
        print(f'Found {file_path}!')
        return
    
    print(f'Could not find {file_path}! Downloading...')
    gdown.download('https://drive.google.com/uc?id='+google_id, file_path)

def create_folder_if_necessary(folder_name: str):
    try:
        os.makedirs(folder_name)
        print(f'Folder "{folder_name}" created...')
    except FileExistsError:
        print(f'Folder "{folder_name}" exists...')
