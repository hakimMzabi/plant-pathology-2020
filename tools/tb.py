"""
Launch tensorboard server and add a tab in the user's favorite browser.
"""
import os

try:
    os.system("start " + os.path.realpath(os.getcwd()) + "\\tools\\tensorboard\\html\\tb.html")
except Exception:
    print("Error: impossible to open the default browser please submit a pull request if you see this error")

try:

    os.system("tensorboard --logdir src/models/logs/tensorboard/fit/")
except Exception:
    print("Error: impossible to launch tensorboard please submit a pull request if you see this error")
