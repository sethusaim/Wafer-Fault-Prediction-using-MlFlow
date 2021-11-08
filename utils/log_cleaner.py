import os
from main_utils import read_params

config = read_params()


def train_log_cleaner(path=config["log_dir"]["train_log_dir"]):
    temp = os.chdir(path)

    temp_path = os.listdir(temp)

    for log in temp_path:
        open(log, "w").close()


def prediction_log_cleaner(path=config["log_dir"]["pred_log_dir"]):
    temp = os.chdir(path)

    temp_path = os.listdir(temp)

    for log in temp_path:
        open(log, "w").close()


if __name__ == "__main__":
    train_log_cleaner()

    os.chdir(config["base"]["project_dir"])

    prediction_log_cleaner()