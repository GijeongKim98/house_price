from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as mse
import numpy as np
import copy


def trainer(model, data, setting, logger):
    train_data = data["train"]
    train_idx = data["train_idx"]

    if setting["n_splits"] <= 1:
        train_count = int(len(train_idx) * (1 - setting["valid_prob"]))
        tr_idx, va_idx = train_idx[:train_count], train_idx[train_count:]

        if setting["model_setting"]["model_name"] == "lightgbm":
            model, loss = lgbm_train(model, tr_idx, va_idx, train_data)
        elif setting["model_setting"]["model_name"] == "xgboost":
            model, loss = xgb_train(model, tr_idx, va_idx, train_data)
        else:
            model, loss = cat_train(model, tr_idx, va_idx, train_data)

        logger.info(f"No K-Fold Result _ loss(rmse) : {loss:.3f}")

        return [model], [loss]

    kf = KFold(n_splits=setting["n_splits"])
    models, losses = [], []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(train_idx)):
        tr_idx, va_idx = train_idx[tr_idx], train_idx[va_idx]
        model_ = copy.deepcopy(model)
        if setting["model_setting"]["model_name"] == "lightgbm":
            model_, loss = lgbm_train(model_, tr_idx, va_idx, train_data)

        elif setting["model_setting"]["model_name"] == "xgboost":
            model_, loss = xgb_train(model_, tr_idx, va_idx, train_data)
        else:
            model_, loss = cat_train(model_, tr_idx, va_idx, train_data)

        logger.info(f"{fold}-Fold  loss(rmse) : {loss:.3f}")
        models.append(model_)
        losses.append(loss)

    return models, losses


def lgbm_train(model, tr_idx, va_idx, train_data):
    train, valid = train_data.loc[tr_idx], train_data.loc[va_idx]
    train_x, train_y = train.drop(columns=["id", "sale_price"]), train["sale_price"]
    valid_x, valid_y = valid.drop(columns=["id", "sale_price"]), valid["sale_price"]

    model.fit(
        train_x,
        train_y,
        eval_set=[(train_x, train_y), (valid_x, valid_y)],
        eval_metric="rmse",
    )

    pred = model.predict(valid_x)
    loss = np.sqrt(mse(valid_y, pred))
    return model, loss


def xgb_train(model, tr_idx, va_idx, train_data):
    train, valid = train_data.loc[tr_idx], train_data.loc[va_idx]
    train_x, train_y = train.drop(columns=["id", "sale_price"]), train["sale_price"]
    valid_x, valid_y = valid.drop(columns=["id", "sale_price"]), valid["sale_price"]

    model.fit(
        train_x,
        train_y,
        eval_set=[(train_x, train_y), (valid_x, valid_y)],
        verbose=False,
    )

    pred = model.predict(valid_x)
    loss = np.sqrt(mse(valid_y, pred))
    return model, loss


def cat_train(model, tr_idx, va_idx, train_data):
    train, valid = train_data.loc[tr_idx], train_data.loc[va_idx]
    train_x, train_y = train.drop(columns=["id", "sale_price"]), train["sale_price"]
    valid_x, valid_y = valid.drop(columns=["id", "sale_price"]), valid["sale_price"]

    model.fit(
        train_x,
        train_y,
        eval_set=[(train_x, train_y), (valid_x, valid_y)],
        verbose=False,
        early_stopping_rounds=10,
    )

    pred = model.predict(valid_x)
    loss = np.sqrt(mse(valid_y, pred))
    return model, loss


# def k_fold_ensemble(models, losses, test, submission, save_path):
#     probs = list(map(lambda x: x / sum(losses), losses))
#     test = test.drop(columns=["id", "sale_price"])

#     sale_price = np.zeros(len(test))

#     for model, prob in zip(models, probs):
#         pred = model.predict(test)
#         sale_price = sale_price + (pred * prob)

#     submission["SalePrice"] = sale_price
#     submission.to_csv(save_path, index=False)

def k_fold_ensemble(models, losses, test, submission, save_path, logger):
    if len(models) != len(losses):
        logger.error("The number of models and losses must be the same.")
        return

    # Normalize the losses to get weights
    total_loss = sum(losses)
    if total_loss == 0:
        logger.error("Total loss is zero, which is not valid for weighting.")
        return
    probs = [total_loss / x for x in losses]
    sum_ = sum(probs)
    probs = [x / sum_ for x in probs]

    logger.info(f"Weights for models: {probs}")

    test = test.drop(columns=["id", "sale_price"], errors='ignore')  # Remove 'id' column if exists

    # Initialize the prediction array
    sale_price = np.zeros(len(test))

    for model, prob in zip(models, probs):
        pred = model.predict(test)
        sale_price += (pred * prob)
        logger.info(f"Model prediction added with weight {prob}")

    submission["SalePrice"] = sale_price

    # Save the submission file
    submission.to_csv(save_path, index=False)
