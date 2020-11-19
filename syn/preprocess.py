import pandas as pd
import numpy as np
import datetime
import gensim
from copy import deepcopy

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

def add_feature_time_place(df, inplace):
    """
    :params df: the input features.
    :params inplace: whether new a col in df.
    if True: return pandas.concat(df, new_features)
    else: return new_features
    :return: new_features or new_df
    """
    def f(x):
        if x > 7 and x <= 13:
            return "0"
        elif x >13 and x <= 19:
            return "1"
        elif (x > 19 and x <= 24) or (x >=0 and x <=1):
            return "2"
        else:
            return "3"
    def g(x):
        if "/" in x:
            return x.split(" / ")[0]
        else:
            return x.split(" Block of ")[1]
    df['TP'] = df['Hour'].apply(f) + df['Address'].apply(g)
    return df

def add_feature_time_district(df, inplace):
    def ff(x):
        if x >= 9 and x <= 17:
            return "1"
        else:
            return "0"

    def gg(x):
        if x == "Monday" or x == "Tuesday" or x == "Wednesday" or x == "Thursday" or x == "Friday":
            return "1"
        else:
            return "0"
    df['TD'] = df['Hour'].apply(ff) + df['DayOfWeek'].apply(gg) + df['PdDistrict']
    return df

def add_feature_dates(df, inplace):
    df['Dates'] = df['Dates'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    df['Year'] = df['Dates'].apply(lambda x: x.year)
    df['Month'] = df['Dates'].apply(lambda x: x.month)
    df['Day'] = df['Dates'].apply(lambda x: x.day)
    df['Hour'] = df['Dates'].apply(lambda x: x.hour)
    df['Minute'] = df['Dates'].apply(lambda x: x.minute)
    df['Special Time'] = df['Minute'].isin([0, 30]).astype(int)
    df.drop('Dates', axis=1, inplace=True)

    df['Weekend'] = df['DayOfWeek'].apply(lambda x: 1 if x == 'Saturday' or x == 'Sunday' else 0)
    df['Night'] = df['Hour'].apply(lambda x: 1 if x > 6 and x < 18 else 0)

    return df

def add_feature_address(df, inplace):
    # add Intersection
    address_list = [address.split(' ') for address in df['Address']]
    address_model = gensim.models.Word2Vec(address_list, min_count=1)
    encoded_address = np.zeros((df.shape[0], 100))
    for i in range(len(address_list)):
        for j in range(len(address_list[i])):
            encoded_address[i] += address_model.wv[address_list[i][j]]
        encoded_address[i] /= len(address_list[i])
    df['Intersection'] = df['Address'].apply(lambda x: 1 if '/' in x else 0)

    # add address embedding
    enc_cols = []
    for i in range(encoded_address.shape[1]):
        enc_cols.append("EncodedAddress{}".format(i))

    enc_add_df = pd.DataFrame(encoded_address, columns=enc_cols)

    combined = pd.concat([df, enc_add_df], axis=1, sort=False)
    combined.drop('Address', axis=1, inplace=True)

    return df

def add_feature_XY(df, inplace):
    X_median = df["X"].median()
    Y_median = df["Y"].median()

    df["X+Y"] = df["X"] + df["Y"]
    df["X-Y"] = df["X"] - df["Y"]

    # df["XY45_1"] = df["X"] * np.cos(np.pi / 4) + df["Y"] * np.sin(np.pi / 4)
    df["XY45_2"] = df["Y"] * np.cos(np.pi / 4) - df["X"] * np.sin(np.pi / 4)

    df["XY30_1"] = df["X"] * np.cos(np.pi / 6) + df["Y"] * np.sin(np.pi / 6)
    df["XY30_2"] = df["Y"] * np.cos(np.pi / 6) - df["X"] * np.sin(np.pi / 6)

    df["XY60_1"] = df["X"] * np.cos(np.pi / 3) + df["Y"] * np.sin(np.pi / 3)
    df["XY60_2"] = df["Y"] * np.cos(np.pi / 3) - df["X"] * np.sin(np.pi / 3)

    df["XY1"] = (df["X"] - df["X"].min()) ** 2 + (df["Y"] - df["Y"].min()) ** 2
    df["XY2"] = (df["X"].max() - df["X"]) ** 2 + (df["Y"] - df["Y"].min()) ** 2
    df["XY3"] = (df["X"] - df["X"].min()) ** 2 + (df["Y"].max() - df["Y"]) ** 2
    df["XY4"] = (df["X"].max() - df["X"]) ** 2 + (df["Y"].max() - df["Y"]) ** 2
    df["XY5"] = (df["X"] - X_median) ** 2 + (df["Y"] - Y_median) ** 2

    df["XY_rad"] = np.sqrt(np.power(df['Y'], 2) + np.power(df['X'], 2))

    pca = PCA(n_components=2).fit(df[["X", "Y"]])
    XYt = pca.transform(df[["X", "Y"]])

    df["XYpca1"] = XYt[:, 0]
    df["XYpca2"] = XYt[:, 1]


    clf = GaussianMixture(n_components=150, covariance_type="diag",
                          random_state=0).fit(df[["X", "Y"]])
    df["XYcluster"] = clf.predict(df[["X", "Y"]])

    return df

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 80)

train_df = pd.read_csv('train.csv', low_memory=False)
test_df = pd.read_csv('test.csv', low_memory=False)
print("read done!")


train_df.drop_duplicates(inplace=True)
print("drop duplicates done!")


train_df.replace({'X': -120.5, 'Y': 90.0}, np.NaN, inplace=True)
test_df.replace({'X': -120.5, 'Y': 90.0}, np.NaN, inplace=True)

imp = SimpleImputer(strategy='mean')
for district in train_df['PdDistrict'].unique():
    train_df.loc[train_df['PdDistrict'] == district, ['X', 'Y']] = imp.fit_transform(train_df.loc[train_df['PdDistrict'] == district, ['X', 'Y']])
    test_df.loc[test_df['PdDistrict'] == district, ['X', 'Y']] = imp.transform(test_df.loc[test_df['PdDistrict'] == district, ['X', 'Y']])
print("replace invalid data done!")


drop_cols = ['Descript', 'Resolution', 'Id']
for col in drop_cols:
    if col in train_df.columns:
        train_df.drop(col, axis=1, inplace=True)
    if col in test_df.columns:
        test_df.drop(col, axis=1, inplace=True)
X = train_df.drop('Category', axis=1)
X_test = test_df
print("drop columns done!")

# tree-based models
y_label = train_df['Category']
le = LabelEncoder()
y_label = le.fit_transform(y_label)
print("make labels done!")


train_length = X.shape[0]

combined = pd.concat([X, X_test], ignore_index=True)
combined = add_feature_dates(combined, True)

# combined = add_feature_time_district(combined, True)
combined = add_feature_time_place(combined, True)

combined = add_feature_address(combined, True)

combined = add_feature_XY(combined, True)

# ## Log Odds
# ### Log Odds - Location

addresses = sorted(train_df['Address'].unique())
categories = sorted(train_df['Category'].unique())

C_counts = train_df.groupby(['Category']).size()
A_C_counts = train_df.groupby(['Address', 'Category']).size()
A_counts = train_df.groupby(['Address']).size()

logodds = {}
logoddsPA = {}

MIN_CAT_COUNTS = 2

default_logodds = np.log(C_counts / len(train_df)) - np.log(1.0 - C_counts / float(len(train_df)))

for addr in addresses:

    PA = A_counts[addr] / float(len(train_df))
    logoddsPA[addr] = np.log(PA) - np.log(1. - PA)
    logodds[addr] = deepcopy(default_logodds)

    for cat in A_C_counts[addr].keys():
        if (A_C_counts[addr][cat] > MIN_CAT_COUNTS) and A_C_counts[addr][cat] < A_counts[addr]:
            PA = A_C_counts[addr][cat] / float(A_counts[addr])
            logodds[addr][categories.index(cat)] = np.log(PA) - np.log(1.0 - PA)

    logodds[addr] = pd.Series(logodds[addr])
    logodds[addr].index = range(len(categories))

new_addresses = sorted(test_df["Address"].unique())
new_A_counts = test_df.groupby("Address").size()

only_new = set(new_addresses + addresses) - set(addresses)
only_old = set(new_addresses + addresses) - set(new_addresses)
in_both = set(new_addresses).intersection(addresses)

for addr in only_new:
    PA = new_A_counts[addr] / float(len(test_df) + len(train_df))
    logoddsPA[addr] = np.log(PA) - np.log(1.0 - PA)
    logodds[addr] = deepcopy(default_logodds)
    logodds[addr].index = range(len(categories))
for addr in in_both:
    PA = (A_counts[addr] + new_A_counts[addr]) / float(len(test_df) + len(train_df))
    logoddsPA[addr] = np.log(PA) - np.log(1.0 - PA)

address_features = combined['Address'].apply(lambda x: logodds[x])
address_features.columns = ['Address Logodds ' + str(x) for x in range(len(address_features.columns))]
combined["logoddsPA"] = combined["Address"].apply(lambda x: logoddsPA[x])

# ### Log Odds - Hour

train_df = add_feature_dates(train_df, True)

hours = sorted(train_df['Hour'].unique())
H_C_counts = train_df.groupby(['Hour', 'Category']).size()
H_counts = train_df.groupby(['Hour']).size()

hour_logodds = {}
hour_logoddsPA = {}

MIN_CAT_COUNTS = 2

default_hour_logodds = np.log(C_counts / len(train_df)) - np.log(1.0 - C_counts / float(len(train_df)))

for hr in hours:

    PH = H_counts[hr] / float(len(train_df))
    hour_logoddsPA[hr] = np.log(PH) - np.log(1. - PH)
    hour_logodds[hr] = deepcopy(default_hour_logodds)

    for cat in H_C_counts[hr].keys():
        if (H_C_counts[hr][cat] > MIN_CAT_COUNTS) and H_C_counts[hr][cat] < H_counts[hr]:
            PH = H_C_counts[hr][cat] / float(H_counts[hr])
            hour_logodds[hr][categories.index(cat)] = np.log(PH) - np.log(1.0 - PH)

    hour_logodds[hr] = pd.Series(hour_logodds[hr])
    hour_logodds[hr].index = range(len(categories))

test_df = add_feature_dates(test_df, True)

new_hours = sorted(test_df["Hour"].unique())
new_H_counts = test_df.groupby("Hour").size()

only_new = set(new_hours + hours) - set(hours)
only_old = set(new_hours + hours) - set(new_hours)
in_both = set(new_hours).intersection(hours)

for hr in only_new:
    PH = new_H_counts[hr] / float(len(test_df) + len(train_df))
    hour_logoddsPA[hr] = np.log(PH) - np.log(1.0 - PH)
    hour_logodds[hr] = deepcopy(default_hour_logodds)
    hour_logodds[hr].index = range(len(categories))
for hr in in_both:
    PH = (H_counts[hr] + new_H_counts[hr]) / float(len(test_df) + len(train_df))
    hour_logoddsPA[hr] = np.log(PH) - np.log(1.0 - PH)

hour_features = combined['Hour'].apply(lambda x: hour_logodds[x])
hour_features.columns = ['Hour Logodds ' + str(x) for x in range(len(hour_features.columns))]
combined["hour logoddsPA"] = combined["Hour"].apply(lambda x: hour_logoddsPA[x])


# ### Log Odds - DayOfWeek

dows = sorted(train_df['DayOfWeek'].unique())

D_C_counts = train_df.groupby(['DayOfWeek', 'Category']).size()
D_counts = train_df.groupby(['DayOfWeek']).size()

dow_logodds = {}
dow_logoddsPA = {}

MIN_CAT_COUNTS = 2

default_dow_logodds = np.log(C_counts / len(train_df)) - np.log(1.0 - C_counts / float(len(train_df)))

for dow in dows:

    PD = D_counts[dow] / float(len(train_df))
    dow_logoddsPA[dow] = np.log(PD) - np.log(1. - PD)
    dow_logodds[dow] = deepcopy(default_dow_logodds)

    for cat in D_C_counts[dow].keys():
        if (D_C_counts[dow][cat] > MIN_CAT_COUNTS) and D_C_counts[dow][cat] < D_counts[dow]:
            PD = D_C_counts[dow][cat] / float(D_counts[dow])
            dow_logodds[dow][categories.index(cat)] = np.log(PD) - np.log(1.0 - PD)

    dow_logodds[dow] = pd.Series(dow_logodds[dow])
    dow_logodds[dow].index = range(len(categories))

new_dows = sorted(test_df["DayOfWeek"].unique())
new_D_counts = test_df.groupby("DayOfWeek").size()

only_new = set(new_dows + dows) - set(dows)
only_old = set(new_dows + dows) - set(new_dows)
in_both = set(new_dows).intersection(dows)

for dow in only_new:
    PD = new_D_counts[dow] / float(len(test_df) + len(train_df))
    dow_logoddsPD[dow] = np.log(PD) - np.log(1.0 - PD)
    dow_logodds[dow] = deepcopy(default_dow_logodds)
    dow_logodds[dow].index = range(len(categories))

for dow in in_both:
    PD = (D_counts[dow] + new_D_counts[dow]) / float(len(test_df) + len(train_df))
    dow_logoddsPA[dow] = np.log(PD) - np.log(1.0 - PD)

dow_features = combined['DayOfWeek'].apply(lambda x: dow_logodds[x])
dow_features.columns = ['DOW Logodds ' + str(x) for x in range(len(dow_features.columns))]
combined["dow logoddsPA"] = combined["DayOfWeek"].apply(lambda x: dow_logoddsPA[x])


print("add features done!")


# encoding
categorical_features = ["DayOfWeek", "PdDistrict", "Intersection", "Special Time", "Intersection", "XYcluster", "TP"]

for col in combined.columns:
    if col in categorical_features:
        oe = OrdinalEncoder()
        combined[col] = oe.fit_transform(combined[col].values.reshape(-1,1))
        combined[col] = combined[col].astype(int)
    elif combined.dtypes[col] == 'object':
        le = LabelEncoder()
        combined[col] = le.fit_transform(combined[col])
print("encode done!")


# save preprocess data
X = combined[:train_length]
X_test = combined[train_length:]


y_label = pd.DataFrame(y_label, columns=['Category'])

X.to_csv('all_train_place.csv', index=False)
X_test.to_csv('all_test_place.csv', index=False)
