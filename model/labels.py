import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

def encoding(df):
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["Category"])

    return le.classes_, df

