import pandas as pd
from typing import Dict
import os


def load_data(setting: dict) -> Dict[str, pd.DataFrame]:
    data = dict()
    data["train"] = pd.read_csv(
        os.path.join(
            os.getcwd(),
            setting["path_setting"]["data_path"],
            setting["name_setting"]["train_name"],
        )
    )
    data["test"] = pd.read_csv(
        os.path.join(
            os.getcwd(),
            setting["path_setting"]["data_path"],
            setting["name_setting"]["test_name"],
        )
    )
    data["submission"] = pd.read_csv(
        os.path.join(
            os.getcwd(),
            setting["path_setting"]["data_path"],
            setting["name_setting"]["submission_name"],
        )
    )

    return data
