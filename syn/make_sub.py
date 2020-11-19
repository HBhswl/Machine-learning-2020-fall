import pandas as pd
import numpy as np

# to make submission from csv

train_df = pd.read_csv('train.csv', low_memory=False)
train_df.drop_duplicates(inplace=True)

y_cats = train_df['Category']
unique_cats = np.sort(y_cats.unique())


lgb_oof_test = pd.read_csv('lgb_oof_test.csv')

sub_df = pd.DataFrame(lgb_oof_test.to_numpy().reshape(-1, 39), columns=unique_cats)

sub_df.index = sub_df.index.set_names(['Id'])
sub_df.reset_index(drop=False, inplace=True)

sub_df.to_csv('lgb_stack.csv', index=False)
