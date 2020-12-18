import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.decomposition import PCA


from features.embedding_xy import remove_outlier_xy, norm_xy, rotate_features_xy, fill_missing_xy
from features.embedding_time import embedding_time
from features.embedding_address import transform_address

from features.preprocess import add_feature_time_place, add_feature_time_district, add_feature_dates
from features.preprocess import add_feature_address, add_feature_XY
from features.preprocess import log_odds_address, log_odds_hour, log_odds_dayofweek, log_odds

from weight.Weight_of_PdDistrict import Weight_of_PdDistrict, Count_of_PdDistrict
from weight.Weight_of_PdDistrict import add_weight_pddistrict, add_weight_pddistrict_label

from model.my_lgb import my_lightgbm_model


def load_data_file():
    df_train = pd.read_csv("train.csv")
    df_test = pd.read_csv("test.csv")

    df_train.index = np.arange(df_train.shape[0])
    df_test.index = np.arange(df_test.shape[0])

    return df_train, df_test

def main():
    df_train, df_test = load_data_file()
    
    # remove outlier
    df_train = remove_outlier_xy(df_train)
    df_test = remove_outlier_xy(df_test)
    df_train, df_test = fill_missing_xy(df_train, df_test)
    print("xy outlier done!")

    # 处理训练集和测试集中的数据差异
    drop_cols = ["Descript", "Resolution", "Id"]
    for col in drop_cols:
        if col in df_train.columns:
            df_train.drop(col, axis=1, inplace=True)
        if col in df_test.columns:
            df_test.drop(col, axis=1, inplace=True)
    
    X = df_train.drop("Category", axis=1)
    X_test = df_test

    # 设置label
    y_label = df_train['Category']
    le = LabelEncoder()
    y_label = le.fit_transform(y_label)

    # 生成特征
    combined = pd.concat([X, X_test], ignore_index=True)
    combined = add_feature_dates(combined, True)
    print("dates done!")

    combined = add_feature_time_district(combined, True)
    combined = add_feature_time_place(combined, True)
    print("time district, time place done!")

    combined = add_feature_address(combined, True)
    print("address done!")    

    combined = add_feature_XY(combined, True)
    print("xy done!")

    combined = log_odds(df_train, df_test, combined, 'PdDistrict', None)
    print("log odds PdDistrict")

    combined = log_odds_address(df_train, df_test, combined)
    print("log odds address done !")

    combined = log_odds_hour(df_train, df_test, combined)
    print("log odds hour done!")

    combined = log_odds_dayofweek(df_train, df_test, combined)
    print("log odds dayofweek done!")    

    # 类别特征数值化
    categorical_features = ["DayOfWeek", "PdDistrict", "Intersection", "Special Time", "Intersection",
                            "XYcluster", "TP", "TD"]

    for col in combined.columns:
        if col in categorical_features:
            oe = OrdinalEncoder()
            combined[col] = oe.fit_transform(combined[col].values.reshape(-1, 1))
            combined[col] = combined[col].astype(int)
        elif combined.dtypes[col] == 'object':
            le = LabelEncoder()
            combined[col] = le.fit_transform(combined[col])

    X = combined[:df_train.shape[0]]
    X_test = combined[df_train.shape[0]:]

    print(X.head(10))
    
    X.to_csv('data/train.csv', index=False)
    X_test.to_csv('data/test.csv', index=False)

if __name__ == "__main__":
    main()
