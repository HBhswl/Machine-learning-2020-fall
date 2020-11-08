import numpy as np
import pandas as pd

def Weight_of_PdDistirct(df, inplace=False):
    """
    :return: (df1, df2, df3),
            df1: [10, 1] the count of incidents in each PdDistrict
            df2: [10, 1] the percent of incidents in each PdDistrict
            df3: [10, 1] the softmax result of the df2.
    """

    assert not inplace, "the inplace should be set False!"

    weight_of_pddistirct = df.groupby("PdDistrict")["Category"].agg(['count'])
    weight_of_pddistirct_percent = weight_of_pddistirct / weight_of_pddistirct.sum()
    weight_of_pddistirct_exp = np.exp(weight_of_pddistirct_percent)
    weight_of_pddistirct_softmax = weight_of_pddistirct_exp / weight_of_pddistirct_exp.sum()

    return weight_of_pddistirct, weight_of_pddistirct_percent, weight_of_pddistirct_softmax

def Count_of_PdDistrict(df, inplace=False):
    """
    :return: pandas.DataFrame
            df1: [10, 39] the count of each kind of incidents in each PdDistrict
    """
    count_of_pddistrict = pd.concat([pd.get_dummies(df["Category"]), df["PdDistrict"]], axis=1).groupby("PdDistrict").agg(['sum'])

    return count_of_pddistrict

if __name__ == "__main__":

    # 读取数据
    df_train = pd.read_csv("train.csv/train.csv", parse_dates=['Dates'])
    df_test = pd.read_csv("test.csv/test.csv", parse_dates=['Dates'])

    # 设置index
    df_train.index = np.arange(df_train.shape[0])
    df_test.index = np.arange(df_test.shape[0])

    print("train set size", df_train.shape)
    print("test set size", df_test.shape)

    df1, df2, df3 = Weight_of_PdDistirct(df_train)
    print(df1)
    print(df2)
    print(df3)

    df4 = Count_of_PdDistrict(df_train)
    print(df4)
