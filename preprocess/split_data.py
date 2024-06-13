import pandas as pd
import numpy as np


def split_data(df: pd.DataFrame) -> dict:
    data = dict()
    train = df[df["sale_price"].isna() == False]
    test = df[df["sale_price"].isna()]

    train_idx = np.array(train.index)
    np.random.shuffle(train_idx)

    data["train"], data["test"], data["train_idx"] = train, test, train_idx

    return data
