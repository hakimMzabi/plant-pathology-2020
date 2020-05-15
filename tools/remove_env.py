"""
Delete a conda environment
"""

import os

if __name__ == "__main__":
    try:
        environment_name = input("Please enter the name of the environment you want to remove: ")
        print(f"Deleting the environment \"{environment_name}\"...")
        os.system(f"conda remove --name {environment_name} --all")
    finally:
        print(f"The environment was successfully deleted!")