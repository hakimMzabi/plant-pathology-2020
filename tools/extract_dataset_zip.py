"""
Delete all training data
"""

import sys
import os
import shutil
import zipfile
from send2trash import send2trash

from src.helper import Helper

if __name__ == "__main__":
    dataset_name = sys.argv[1]
    helper = Helper()
    dir_name = f"./data/{dataset_name}/"
    print(f"Starting dataset zip extraction in {dir_name}...")
    try:
        helper.create_dir(dir_name)
    except Exception:
        print("Error: Could not create directory.")
    zip_name = f"{dataset_name}.zip"
    if not os.path.isfile(zip_name):
        print(f"Error: {zip_name} is not a file.\nAborted dataset zip extraction.")
    else:
        # move zip to dir
        full_new_path = f"{dir_name}/{zip_name}"
        shutil.move(f"{zip_name}", f"{full_new_path}")
        print(f"Success: {zip_name} moved to {full_new_path}.")
        # extract zip in dir
        with zipfile.ZipFile(f'{full_new_path}', 'r') as zipObj:
            # Extract all the contents of zip file in current directory
            zipObj.extractall(f"{dir_name}")
        print(f"Success: {full_new_path} was successfully uncompressed!")
        send2trash(f"{full_new_path}")
        print(f"Success: Successfully removed {full_new_path} at the dataset folder!")

