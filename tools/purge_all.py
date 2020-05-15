"""
Delete all training data
"""

import sys
from send2trash import send2trash

import os, re


def purge_verification(name):
    verif = input(f"Are you sure you want to purge {name} ? (y/n) : ")
    if verif != "y" and verif != "n":
        print("Only (y) and (n) are possible options")
        purge_verification(name)
    else:
        return verif


def purge(dir, pattern, name):
    deleted = False
    for f in os.listdir(dir):
        if re.search(pattern, str(f)):
            if not deleted:
                deleted = True
            file_to_delete = f"{dir}/{f}"
            send2trash(file_to_delete)
            print(f"Deleted {file_to_delete}.")
    if deleted:
        print(f"Success: Successfully deleted {name}.")
    else:
        print(f"Error: No {name} found to purge.")


def purge_on_verif(name, dir, pattern):
    if purge_verification(name) == "y":
        purge(dir, pattern, name)
        if name == "playground.py":
            f = open("src/playground.py","w")
            f.close()


def purge_logs():
    purge_on_verif("training data logs", "src/models/logs", r".log")
    purge_on_verif("tensorboard data", "src/models/logs/tensorboard/fit", r"[0-9]{8}-[0-9]{6}")
    purge_on_verif("model data", "src/models/responses", r".h5")
    purge_on_verif("checkpoints", "src/models/checkpoints", r".ckpt")
    purge_on_verif("main checkpoint", "src/models/checkpoints", r"checkpoint")
    purge_on_verif("playground.py", "src", r"playground.py")


if __name__ == "__main__":
    print(f"Starting training data purge...")
    purge_logs()
