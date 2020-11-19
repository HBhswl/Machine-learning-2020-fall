# count the size of different features

import pandas as pd

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 25)

train_df = pd.read_csv('train.csv', low_memory=False)

train_df['Address 1'] = train_df['Address'].apply(lambda x: x.split(" / ")[0] if '/' in x else x.split(" Block of ")[1])
f1 = train_df['Address 1'].unique()
print(len(f1))

train_df['Address 2'] = train_df['Address'].apply(lambda x: x if '/' in x else x.split(" Block of ")[1])
f2 = train_df['Address 2'].unique()
print(len(f2))

train_df['Address 3'] = train_df['Address'].apply(lambda x: x.split(" / ")[0] if '/' in x else x)
f3 = train_df['Address 3'].unique()
print(len(f3))