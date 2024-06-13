import json
import os
from typing import List, Tuple, Dict
import random
import numpy as np
import pandas as pd
import logging
import logging.config
from datetime import datetime


def load_setting() -> dict:
    path = os.path.join(os.getcwd(), "setting.json")
    with open(path) as json_file:
        total_setting = json.load(json_file)
    total_setting["path_setting"]["now_path"] = os.getcwd()
    return total_setting


def load_feature_info(path):
    with open(path) as col_file:
        feature_info = json.load(col_file)
    return feature_info


def load_category2score(path):
    with open(path) as col_file:
        category2score = json.load(col_file)
    return category2score


def get_logger(file_name: str):
    # Set logging
    logger_conf = {  # only used when 'user_wandb==False'
        "version": 1,
        "formatters": {  # formatters => Basic 사용
            "basic": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"}
        },  # 로그 작성 시간, 작성 이름, 레벨(Info, Debug 등), 로그메세지
        "handlers": {  # 로그 출력 방식
            "console": {  # 터미널
                "class": "logging.StreamHandler",
                "level": "DEBUG",
                "formatter": "basic",
                "stream": "ext://sys.stdout",
            },
            "file_handler": {  # 파일
                "class": "logging.FileHandler",
                "level": "DEBUG",
                "formatter": "basic",
                "filename": f"logs/{file_name}.log",
            },
        },
        "root": {"level": "DEBUG", "handlers": ["console", "file_handler"]},
    }

    logging.config.dictConfig(logger_conf)
    logger = logging.getLogger()
    return logger


def check_dir(path_setting: dict):
    for dir_key, dir_name in path_setting.items():
        if dir_key == "data" or dir_key == "now_path":
            continue
        if not os.path.isdir(os.path.join(path_setting["now_path"], dir_name)):
            os.mkdir(os.path.join(path_setting["now_path"], dir_name))


def init(setting: dict) -> tuple:
    # check dir and make dir
    check_dir(setting["path_setting"])

    # now time
    setting["now_times"] = datetime.now().strftime("%Y%m%d_%H%M%S")

    # save file path
    setting["name_setting"]["save_model"] = (
        setting["model_setting"]["model_name"] + "_" + setting["now_times"]
    )
    setting["name_setting"]["save_submission"] = (
        setting["model_setting"]["model_name"] + "_" + setting["now_times"]
    )

    # Set Feature info
    data_dir_path = os.path.join(
        setting["path_setting"]["now_path"], setting["path_setting"]["data_path"]
    )

    setting["feature_info"] = load_feature_info(
        os.path.join(data_dir_path, setting["name_setting"]["column"])
    )

    # Model and Preprocessing
    if setting["model_setting"]["model_name"] == "catboost":
        setting["is_catboost"] = True
    else:
        setting["is_catboost"] = False

    if not setting["is_catboost"] or setting["is_score"]:
        setting["category2score"] = load_category2score(
            os.path.join(data_dir_path, setting["name_setting"]["category2score"])
        )

    # Logger Config
    logger = get_logger(setting["now_times"])

    # Set Seed number
    seed = setting["seed"]
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    return setting, logger


def get_save_submission_path(setting):
    return os.path.join(
        setting["path_setting"]["now_path"],
        setting["path_setting"]["save_submission"],
        setting["name_setting"]["save_submission"] + ".csv",
    )
