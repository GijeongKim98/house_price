from utils.utils import load_setting, init, get_save_submission_path
from preprocess import *
from model.model import load_model
from trainer.trainer import trainer
from trainer.trainer import k_fold_ensemble


def main():
    # Load Setting (setting.json)
    setting = load_setting()

    # Initialization setting and logger
    setting, logger = init(setting)
    logger.info("Initialize setting, logger")

    # Load Data
    data = load_data(setting)
    logger.info("Load Data : Train, Test, Submission file")
    submission = data["submission"]

    # Preprocess : missing value, etc...
    feature, setting["feature_info"] = preprocess_data(data, setting)
    logger.info("Preprocess Data : Train + Test ")

    data = split_data(feature)
    logger.info("Split data : Train / Test")

    # Load Model
    model = load_model(setting["model_setting"], feature_info=setting["feature_info"])
    logger.info(f"load model : model_name [{setting['model_setting']['model_name']}]")

    # Train Model
    logger.info(f"[{setting['model_setting']['model_name']}] Start Training ...")
    models, losses = trainer(model, data, setting, logger)
    logger.info(
        f"[{setting['model_setting']['model_name']}] Train Complete. Mean_Loss : {sum(losses)/len(losses):.3f}"
    )

    # K-Fold Ensemble & Save Submission
    logger.info("Start ensemble..(k-fold) & Save Submission file")
    save_path = get_save_submission_path(setting)
    k_fold_ensemble(models, losses, data["test"], submission, save_path, logger)
    logger.info(f"Submission file saved to {setting['name_setting']['save_submission']}.csv")
    
    return


if __name__ == "__main__":
    main()
