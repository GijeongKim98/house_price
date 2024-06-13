from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

def load_model(model_setting, feature_info=None):
    model_name = model_setting["model_name"].lower()
    model_params = model_setting[f"{model_name}_parameters"]
    
    if model_name == "lightgbm":
        model = LGBMRegressor(**model_params)
    
    elif model_name == "catboost":
        model = CatBoostRegressor(cat_features=feature_info["category1"], **model_params)
    
    else:
        model = XGBRegressor(**model_params)
    
    return model
        