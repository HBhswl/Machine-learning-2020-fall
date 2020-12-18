import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer

# 删除掉训练集中的异常数据，并且将训练集和测试集中的经纬度进行均一化操作
def remove_outlier_xy(df, inplace=True):
    assert inplace, "to remove outlier, the inplace should be set True!"
    # df = df[df["Y"] < 80]
    df.replace({'X': -120.5, 'Y': 90.0}, np.NaN, inplace=True)
    return df

def fill_missing_xy(train_df, test_df):
    imp = SimpleImputer(strategy='mean')
    for district in train_df['PdDistrict'].unique():
        train_df.loc[train_df['PdDistrict'] == district, ['X', 'Y']] = \
            imp.fit_transform(train_df.loc[train_df['PdDistrict'] == district, ['X', 'Y']])
        test_df.loc[test_df['PdDistrict'] == district, ['X', 'Y']] = \
            imp.transform(test_df.loc[test_df['PdDistrict'] == district, ['X', 'Y']])
    return train_df, test_df

# 对xy进行标准化
def norm_xy(df1, df2, inplace=True):
    # df1 是训练集数据，df2是测试集数据
    tmp_x = pd.concat([df1["X"], df2["X"]], axis=0)
    tmp_y = pd.concat([df1["Y"], df2["Y"]], axis=0)

    x_mean, x_std = tmp_x.mean(), tmp_x.std()
    y_mean, y_std = tmp_y.mean(), tmp_y.std()

    df1["X-norm"] = (df1["X"] - x_mean)/x_std
    df2["X-norm"] = (df2["X"] - x_mean)/x_std
    df1["Y-norm"] = (df1["Y"] - y_mean)/y_std
    df2["Y-norm"] = (df2["Y"] - y_mean)/y_std

    if inplace:
        return df1, df2
    else:
        return df1[["X-norm", "Y-norm"]], df2[["X-norm", "Y-norm"]]

# 对xy计算一些features
def rotate_features_xy(df, inplace=True):
    assert "X-norm" in list(df.columns) and "Y-norm" in list(df.columns), \
        "you should first norm xy!"
    cols = [
        "rot30_X", "rot30_Y",
        "rot45_X", "rot45_Y",
        "rot60_X", "rot60_Y",
        "xy", "radius_xy"
    ]
    df["rot30_X"] = (np.cos(np.pi/6)) * df["X-norm"] + (np.sin(np.pi/6)) * df["Y-norm"]
    df["rot30_Y"] = (np.cos(np.pi/6)) * df["Y-norm"] - (np.sin(np.pi/6)) * df["X-norm"]

    df["rot45_X"] = (np.cos(np.pi/4)) * df["X-norm"] + (np.sin(np.pi/4)) * df["Y-norm"]
    df["rot45_Y"] = (np.cos(np.pi/4)) * df["Y-norm"] - (np.sin(np.pi/4)) * df["X-norm"]

    df["rot45_X"] = (np.cos(np.pi/3)) * df["X-norm"] + (np.sin(np.pi/3)) * df["Y-norm"]
    df["rot45_Y"] = (np.cos(np.pi/3)) * df["Y-norm"] - (np.sin(np.pi/3)) * df["X-norm"]

    df["xy"] = df["X-norm"] * df["Y-norm"]
    df["radius_xy"] = np.sqrt(np.power(df["X-norm"], 2) + np.power(df["Y-norm"], 2))

    if inplace:
        return df
    else:
        return df[cols]

if __name__ == "__main__":
    # 读取数据
    df_train = pd.read_csv("train.csv/train.csv", parse_dates=['Dates'])
    df_test = pd.read_csv("test.csv/test.csv", parse_dates=['Dates'])

    # 设置index
    df_train.index = np.arange(df_train.shape[0])
    df_test.index = np.arange(df_test.shape[0])

    print("train set size", df_train.shape)
    print("test set size", df_test.shape)

    # 删除异常数据
    df_train = remove_outlier_xy(df_train, inplace=True)
    df_test = remove_outlier_xy(df_test, inplace=True)

    # 标准化
    df_train, df_test = norm_xy(df_train, df_test, inplace=True)

    # xy的特征挖掘
    df_train = rotate_features_xy(df_train, inplace=True)
    df_test = rotate_features_xy(df_test, inplace=True)

    print(df_train.head())
