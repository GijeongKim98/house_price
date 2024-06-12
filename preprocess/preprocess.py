import pandas as pd
import numpy as np
from scipy.stats import skew
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.preprocessing import MultiLabelBinarizer
from typing import Dict, Tuple


def process_miss_value_ca1(df: pd.DataFrame) -> pd.DataFrame:
    # exterior_1st & exterior_2nd:
    miss_exter_idx = df[df["exterior_1st"].isna() & df["exterior_2nd"].isna()].index

    df.loc[miss_exter_idx, "exterior_1st"] = "MetalSd"

    df.loc[miss_exter_idx, "exterior_2nd"] = "MetalSd"

    # mas_vnr_type, mas_vnr_area
    no_vnr_type_idx = df[df["mas_vnr_type"].isna()].index

    no_vnr_area_idx = df[
        (df["mas_vnr_area"] == 0) | (df["mas_vnr_area"].isna() == True)
    ].index

    miss_vnr_type_idx = list(set(no_vnr_type_idx) - set(no_vnr_area_idx))

    miss_vnr_area_idx = list(set(no_vnr_area_idx) - set(no_vnr_type_idx))

    df.loc[miss_vnr_type_idx, "mas_vnr_type"] = "BrkFace"

    gb = df.groupby("mas_vnr_type").agg({"mas_vnr_area": "mean"})

    for idx in miss_vnr_area_idx:
        df.loc[idx, "mas_vnr_area"] = gb.loc[df.loc[idx, "mas_vnr_type"]].values[0]

    df["mas_vnr_area"] = df["mas_vnr_area"].fillna(0)

    # misc_feature
    miss_misc_feat_idx = df[(df["misc_feature"].isna()) & (df["misc_val"] != 0)].index
    df.loc[miss_misc_feat_idx, "misc_feature"] = "Gar2"

    # ms_zoning : mode over neighborhood
    df.loc[[1915, 2216, 2250], "ms_zoning"] = "RM"
    df.loc[2904, "ms_zoning"] = "RL"

    # sale_type mode
    df.loc[df[df["sale_type"].isna()].index, "sale_type"] = "WD"

    # utilities
    df.loc[df[df["utilities"].isna()].index, "utilities"] = "AllPub"

    return df


def process_miss_value_bsmt(df: pd.DataFrame) -> pd.DataFrame:
    # bsmt columns : ["bsmt_qual","bsmt_cond","bsmt_exposure","bsmt_fin_type1","bsmt_fin_type2",
    #                 "bsmt_full_bath","bsmt_half_bath", "bsmt_fin_sf1", "bsmt_fin_sf2","bsmt_un_sf",
    #                 "total_bsmt_sf"]

    non_bsmt_idx = df[(df["bsmt_cond"].isna()) & (df["bsmt_exposure"].isna())].index

    miss_bstm_cond_idx = list(set(df[df["bsmt_cond"].isna()].index) - set(non_bsmt_idx))

    miss_bstm_qual_idx = list(set(df[df["bsmt_qual"].isna()].index) - set(non_bsmt_idx))

    miss_bstm_exposure_idx = list(
        set(df[df["bsmt_exposure"].isna()].index) - set(non_bsmt_idx)
    )

    miss_bstm_fin_type2_idx = list(
        set(df[df["bsmt_fin_type2"].isna()].index) - set(non_bsmt_idx)
    )

    # bsmt_fin_type2
    df.loc[miss_bstm_fin_type2_idx, "bsmt_fin_type2"] = "ALQ"

    # bsmt_exposure
    df.loc[miss_bstm_exposure_idx, "bsmt_exposure"] = "No"

    # bsmt_qual
    df.loc[miss_bstm_qual_idx, "bsmt_qual"] = "TA"

    # bsmt_cond
    df.loc[miss_bstm_cond_idx, "bsmt_cond"] = "TA"

    # others
    other_bsmt_cols = [
        "bsmt_full_bath",
        "bsmt_half_bath",
        "bsmt_fin_sf1",
        "bsmt_fin_sf2",
        "bsmt_un_sf",
        "total_bsmt_sf",
    ]
    df[other_bsmt_cols] = df[other_bsmt_cols].fillna(0)

    return df


def process_miss_value_garage(df: pd.DataFrame) -> pd.DataFrame:
    # garage_cols = ["garage_type","garage_finish","garage_qual","garage_cond","garage_cars","garage_yr_built", "garage_area"]
    # none garage data : total 157 => 158(add 2576)
    # miss data  : total 2 // idx : 2576 2126

    # garage_type / idx 2576 => non garage data
    df.loc[2576, "garage_type"] = np.nan

    # garage_finish
    df.loc[2126, "garage_finish"] = "Unf"

    # garage_qual
    df.loc[2126, "garage_qual"] = "TA"

    # garage_cond
    df.loc[2126, "garage_cond"] = "TA"

    # garage_yr_built
    df.loc[2126, "garage_yr_built"] = 1947

    # garage_cars, garage_area
    others = ["garage_cars", "garage_area"]
    df[others] = df[others].fillna(0)

    return df


def process_miss_value_ca2(df: pd.DataFrame) -> pd.DataFrame:
    # electrical
    df["electrical"] = df["electrical"].fillna("SBrkr")

    # kitchen_qual
    df["kitchen_qual"] = df["kitchen_qual"].fillna("TA")

    # functional
    df["functional"] = df["functional"].fillna("Typ")

    return df


def process_miss_value_integer(df: pd.DataFrame) -> pd.DataFrame:
    # lot_frontage
    gb = df.groupby("neighborhood").agg({"lot_frontage": "mean"})
    miss_lot_frontage_idxes = df[df["lot_frontage"].isna()].index

    for idx in miss_lot_frontage_idxes:
        df.loc[idx, "lot_frontage"] = gb.loc[df.loc[idx, "neighborhood"]].values[0]

    return df


def get_miss_values(df: pd.DataFrame) -> None:
    total_count = len(df)
    for col in df.columns:
        miss_count = len(df[df[col].isna()])
        if miss_count > 0:
            print(f"[{col}] : {miss_count} / {total_count}")
    print("\n\n")


def process_miss_data(feature: pd.DataFrame) -> pd.DataFrame:
    # Missing Value - Category1:
    feature = process_miss_value_ca1(feature)

    # Missing Value - BSMT:
    feature = process_miss_value_bsmt(feature)

    # Missing Value - Garage:
    feature = process_miss_value_garage(feature)

    # Missing Value - Category2:
    feature = process_miss_value_ca2(feature)

    # Missing Value - Integer:
    feature = process_miss_value_integer(feature)

    return feature


def feature_engineering(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict, dict]:
    column_category = [
        "category1",
        "category2",
        "integer1",
        "integer2",
        "integer3",
        "time_data",
    ]
    delete_feature = {col_info: [] for col_info in column_category}
    generate_feature = {col_info: [] for col_info in column_category}

    # fence_qual => is_fence, delete = True
    df["is_fence"] = df["fence_qual"].apply(lambda x: 0 if x == np.nan else 1)
    delete_feature["category2"].append("fence_qual")
    generate_feature["category1"].append("is_fence")

    # pool_qual, pool_area => is_pool, delete = True
    df["is_pool"] = df["pool_area"].apply(lambda x: 1 if x > 0 else 0)
    delete_feature["category2"].append("pool_qual")
    delete_feature["integer2"].append("pool_area")
    generate_feature["category1"].append("is_pool")

    # functional -> is_functional, delete = True
    df["is_functional"] = df["functional"].apply(lambda x: 0 if x == "Typ" else 1)
    delete_feature["category2"].append("functional")
    generate_feature["category1"].append("is_functional")

    # is_2f
    df["is_2f"] = df["2nd_flr_sf"].apply(lambda x: 1 if x > 0 else 0)
    generate_feature["category1"].append("is_2f")

    # total_bsmt_bath
    df["total_bsmt_bath"] = df["bsmt_half_bath"] * 0.5 + df["bsmt_full_bath"]
    generate_feature["integer1"].append("total_bsmt_bath")

    # total_bath
    df["total_bath"] = df["half_bath"] * 0.5 + df["full_bath"]
    generate_feature["integer1"].append("total_bath")

    # is_bsmt_bath
    df["is_bsmt_bath"] = df["total_bsmt_bath"].apply(lambda x: 1 if x > 0 else 0)
    generate_feature["category1"].append("is_bsmt_bath")

    # all_bath
    df["all_bath"] = df["total_bath"] + df["total_bsmt_bath"]
    generate_feature["integer1"].append("all_bath")

    # is_remod
    df["is_remod"] = df.apply(
        lambda x: 0 if x["year_built"] == x["year_remod_add"] else 1, axis=1
    )
    generate_feature["category1"].append("is_remod")

    # diff_built_sold
    df["diff_built_sold"] = df["yr_sold"] - df["year_built"]
    generate_feature["time_data"].append("diff_built_sold")

    # diff_remod_sold
    df["diff_remod_sold"] = df["yr_sold"] - df["year_remod_add"]
    generate_feature["time_data"].append("diff_remod_sold")

    # diff_grg_sold
    # garage_yr_built - process miss_value(non garage data)
    df["garage_yr_built"] = df["garage_yr_built"].fillna(1960)
    df["diff_grg_sold"] = df["yr_sold"] - df["garage_yr_built"]
    generate_feature["time_data"].append("diff_grg_sold")

    # year_built_10
    df["year_built_10"] = df["year_built"].apply(lambda x: (x // 10) * 10)
    generate_feature["time_data"].append("year_built_10")

    del_cols = set()
    for _, list_ in delete_feature.items():
        del_cols = del_cols.union(set(list_))

    del_cols = list(del_cols)

    df = df.drop(columns=del_cols)

    return df, delete_feature, generate_feature


def revise_feature(
    feature_info: dict, delete_feature: dict, generate_feature: dict, cols: list
) -> dict:
    features = []
    keys = list(feature_info.keys())
    for col_category in keys:
        if col_category in delete_feature.keys():
            feature_info[col_category] = list(
                set(feature_info[col_category] + generate_feature[col_category])
                - set(delete_feature[col_category])
            )
            features += feature_info[col_category]
        else:
            del feature_info[col_category]

    return feature_info


def process_total_data(df, setting):
    # get_miss_values(df)
    # Preprocess category - order
    category2 = setting["feature_info"]["category2"]
    if not setting["is_catboost"] or setting["is_score"]:
        # except overall_qual , overall_cond
        scorable_category2 = set(setting["category2score"].keys())
        cols = list(set(category2).intersection(scorable_category2))
        for col in cols:
            df[col] = df[col].fillna("None")  # bsmt or garage or fireplace
            df[col] = df[col].map(setting["category2score"][col])

        skew_features = (
            df[category2].apply(lambda x: skew(x)).sort_values(ascending=False)
        )

        high_skew = skew_features[skew_features > 0.5]
        skew_index = high_skew.index

        for i in skew_index:
            df[i] = boxcox1p(df[i], boxcox_normmax(df[i] + 1, method="mle"))

    else:
        category2 = list(set(category2) - {"overall_qual", "overall_cond"})
        setting["feature_info"]["category1"] += category2

    # Preprocess category - category
    category1 = setting["feature_info"]["category1"]
    if setting["is_catboost"]:
        df[category1] = df[category1].fillna("None")
        for col in category1:
            if df[col].dtype != np.dtype("object"):
                df[col] = df[col].astype(str)

    else:
        one_hot_en_cols = set(category1) - {
            "is_fence",
            "is_functional",
            "is_2f",
            "is_pool",
            "is_bsmt_bath",
            "is_remod",
            "central_air",
        }
        multi_hot_en_cols = {"exterior_1st", "exterior_2nd", "condition1", "condition2"}
        one_hot_en_cols -= multi_hot_en_cols
        df["ms_subclass"] = df["ms_subclass"].astype(str)
        df["central_air"] = df["central_air"].apply(lambda x: 1 if x == "Y" else 0)
        df = pd.get_dummies(df, columns=list(one_hot_en_cols))

        # exterior_1st, exterior_2nd
        df["exterior"] = df.apply(
            lambda x: [x["exterior_1st"], x["exterior_2nd"]], axis=1
        )
        df["exterior"] = df["exterior"].apply(lambda x: [x[0]] if x[0] == x[1] else x)
        mlb = MultiLabelBinarizer()
        mlb.fit(df["exterior"])
        new_ext_col_name = ["exterior_%s" % c for c in mlb.classes_]
        mlb_df = pd.DataFrame(mlb.transform(df["exterior"]), columns=new_ext_col_name)
        df = pd.concat([df, mlb_df], axis=1)

        # condition1, condition2
        df["condition"] = df.apply(lambda x: [x["condition1"], x["condition2"]], axis=1)
        df["condition"] = df["condition"].apply(lambda x: [x[0]] if x[0] == x[1] else x)
        mlb = MultiLabelBinarizer()
        mlb.fit(df["condition"])
        new_ext_col_name = ["condition_%s" % c for c in mlb.classes_]
        mlb_df = pd.DataFrame(mlb.transform(df["condition"]), columns=new_ext_col_name)
        df = pd.concat([df, mlb_df], axis=1)

        # drop  // feature info
        df = df.drop(columns=list(multi_hot_en_cols))
        setting["feature_info"]["category1"] = list(
            set(setting["feature_info"]["category1"]) - multi_hot_en_cols
        )
        setting["feature_info"]["category1"].append("exterior")
        setting["feature_info"]["category1"].append("condition")

    # Preprocess - time_data
    time_category = ["mo_sold", "year_built_10"]

    for tc in time_category:
        df[tc] = df[tc].astype(str)
        setting["feature_info"]["category1"].append(tc)

    if not setting["is_catboost"]:
        df = pd.get_dummies(df, columns=time_category)

    setting["feature_info"]["time_data"] = list(
        set(setting["feature_info"]["time_data"]) - set(time_category)
    )

    skew_features = (
        df[setting["feature_info"]["time_data"]]
        .apply(lambda x: skew(x))
        .sort_values(ascending=False)
    )

    high_skew = skew_features[skew_features > 0.5]
    skew_index = high_skew.index

    for i in skew_index:
        df[i] = boxcox1p(df[i], boxcox_normmax(df[i] + 2, method="mle"))

    # Preprocess - integer data
    # integer = integer1 + integer2 + integer3
    # high skew => boxcox
    integer = (
        setting["feature_info"]["integer1"]
        + setting["feature_info"]["integer2"]
        + setting["feature_info"]["integer3"]
    )
    skew_features = df[integer].apply(lambda x: skew(x)).sort_values(ascending=False)

    high_skew = skew_features[skew_features > 0.5]
    skew_index = high_skew.index

    for i in skew_index:
        df[i] = boxcox1p(df[i], boxcox_normmax(df[i] + 1, method="mle"))

    return df


def preprocess_data(data: Dict[str, pd.DataFrame], setting: dict) -> pd.DataFrame:
    feature = pd.concat([data["train"], data["test"]], ignore_index=True)

    # Rename column:
    feature = feature.rename(columns=setting["feature_info"]["columns_name"])

    # Process Missing Value:
    feature = process_miss_data(feature)

    # Feature Engineering
    feature, delete_feature, generate_feature = feature_engineering(feature)

    # process Feature Info
    setting["feature_info"] = revise_feature(
        setting["feature_info"], delete_feature, generate_feature, feature.columns
    )

    # Preprocess
    feature = process_total_data(feature, setting)

    # get_miss_values(feature)

    print(feature.shape)

    return feature, setting["feature_info"]
