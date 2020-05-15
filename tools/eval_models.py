"""
Evaluate n models from a process_name
"""
import pprint
import sys
from src.helper import Helper

if __name__ == "__main__":
    helper = Helper()
    arg_len = len(sys.argv)
    arg_len_is_valid = arg_len != 3 or arg_len != 2

    if not arg_len_is_valid:
        print("Usage:\n    python -m tools.evaluate_models [nb_of_models] [process_name]\n    or\n    python -m tools.evaluate_models [process_name]")
        exit(1)
    if arg_len == 2:
        nb_of_models = int(sys.argv[1])
        pprint.pprint(helper.evaluate_models(nb_of_models))
    elif arg_len == 3:
        nb_of_models = int(sys.argv[1])
        process_name = sys.argv[2]
        pprint.pprint(helper.evaluate_models(nb_of_models, process_name))
