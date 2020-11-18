import pandas as pd
import numpy as np

def transform_address(df):
    df['Block'] = df['Address'].str.contains('block', case=False)
    df['Block'] = df['Block'].map(lambda x: 1 if  x == True else 0)
    
    df = pd.get_dummies(data=df, columns=['PdDistrict'], drop_first = True)
    df.drop(columns=["Address"], inplace=True)
    return df
