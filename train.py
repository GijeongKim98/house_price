from utils.utils import load_setting, init
from preprocess import *


def main():
    # Load Setting (setting.json)
    setting = load_setting()

    # Initialization setting and logger
    setting, logger = init(setting)
    # logger.info("Initialize setting, logger")

    # Load Data
    data = load_data(setting)
    # logger.info("Load Data : Train, Test, Submission file")

    # Preprocess : missing value, etc...
    feature, setting["feature_info"] = preprocess_data(data, setting)
    # logger.info("Preprocess Data : Train + Test ")

    # Split Data : feature => train, test

    # Setting k_fold

    # Load Model

    # Train Model

    # Validation Model

    # Save Model

    # Save Submission

    return


if __name__ == "__main__":
    main()
