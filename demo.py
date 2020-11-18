import numpy as np
import pandas as pd

from features.embedding_xy import remove_outlier_xy, norm_xy, rotate_features_xy
from features.embedding_time import embedding_time
from features.embedding_address import transform_address

from weight.Weight_of_PdDistrict import Weight_of_PdDistrict, Count_of_PdDistrict
from weight.Weight_of_PdDistrict import add_weight_pddistrict, add_weight_pddistrict_label

from model.my_lgb import my_lightgbm_model


def load_data_file():
    df_train = pd.read_csv("train.csv", parse_dates=["Dates"])
    df_test = pd.read_csv("test.csv", parse_dates=["Dates"])

    df_train.index = np.arange(df_train.shape[0])
    df_test.index = np.arange(df_test.shape[0])

    return df_train, df_test, df_test.copy()

def main():
    df_train, df_test, df_test2 = load_data_file()

    # remove outlier
    df_train = remove_outlier_xy(df_train)
    df_test = remove_outlier_xy(df_test)

    # weight of pddistrict
    _, weight_of_pddistrict, weight_of_pddistrict_softmax = Weight_of_PdDistrict(df_train)
    count_of_pddistrict_label = Count_of_PdDistrict(df_train)

    # add weight
    df_train = add_weight_pddistrict_label(df_train, count_of_pddistrict_label)

    # xy features
    df_train, df_test = norm_xy(df_train, df_test, inplace=True)
    df_train = rotate_features_xy(df_train, inplace=True)
    df_test = rotate_features_xy(df_test, inplace=True)

    # time features
    df_train = embedding_time(df_train)
    df_test = embedding_time(df_test)

    # address features
    df_train = transform_address(df_train)
    df_test = transform_address(df_test)

    print(df_train.head(10))
    print(df_test.head(10))

    my_lightgbm_model(df_train, df_test, df_test2)

if __name__ == "__main__":
    main()
