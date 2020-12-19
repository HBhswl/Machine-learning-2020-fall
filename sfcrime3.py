import pandas as pd
from shapely.geometry import Point
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
from matplotlib import cm
import urllib.request
import shutil
import zipfile
import os
import re
import contextily as ctx
# import geoplot as gplt
import lightgbm as lgb
import math
import time
from gensim.models import Doc2Vec
import nltk
nltk.data.path.append("/home/wpc/nltk_data/")
from nltk.tokenize import word_tokenize


# use dict to change parameter to vector, the dim depend on the longest parameter
# eg: address, descript, resolution
# data --> target pd.DataFrame 
# para --> parameter to  convert
# splitVar --> char to split, default is space(' ')
# dataDict --> vector value dict generate by train , and used in both train and test
# dimCount --> lenth of the vector a single parameter converted to 
def parameterToVector(data, para, splitVar=' ', dataDict={}, dimCount = 0):
    # data = pd.read_csv('../input/train.csv')
    dataList = []
    if(dimCount == 0):
        for it, item in enumerate(np.unique(data[para].values).tolist()):
            # clear the data string, delete / in address
            item = item.replace('/', '')
            part = len(item.split(splitVar))
            dataList.extend(item.split(splitVar))
            if(part > dimCount):
                dimCount = part

    # build dict for all data
    if(len(dataDict) == 0):
        dataDict = {}
        for it, item in enumerate(np.unique(dataList)):
            # print(it,item)
            dataDict[item] = it + 1

    paraNarray = np.zeros(shape=(data.shape[0], dimCount))

    for it, item in enumerate(data[para].values):
        # print(it,item)
        item = item.replace('/', '')
        itemlist = item.split(splitVar)
        addv = [-1]*dimCount
        for p, ad in enumerate(itemlist):
            if(ad in dataDict.keys()):
                addv[p] = dataDict[ad]
            else:
                addv[p] = -1
        paraNarray[it] = addv

    paraColumn = []
    for it in range(dimCount):
        paraColumn.append(para+str(it))
    paraPD = pd.DataFrame(data=paraNarray, columns=paraColumn)
    data = pd.concat([data, paraPD], axis=1,
                     ignore_index=False)
    # print(data)
    # print(dimCount)
    return data, dataDict


def addr2list(addr):
    addr = addr.strip()
    blocknum = -1
    regionName = ''
    regionType = ''
    # if there is block
    if (' of ' in addr):
        al = addr.split(' of ')
        blocknum = int(al[0].split(' ')[0])
        regionlist = al[1].split(' ')
        regionName = ".".join(regionlist[:-1])
        regionType = "".join(regionlist[-1:])
    else:
        al = addr.split(' ')
        regionName = ".".join(al[:-1])
        regionType = "".join(al[-1:])
    return [blocknum, regionName, regionType]

def buildAddrDict(data):
    # data = pd.read_csv('../input/train.csv')
    dataList = []
    splitVar = '/'
    for it,item in enumerate(data['Address'].unique()):
        dataList.extend(item.split(splitVar))
    # make dataList unique
    addrSet = set(dataList)

    # build dict for address
    addrDict = {}
    typeDictValue = 1
    NameDictValue = 100
    for item in addrSet:
        adl = addr2list(item)
        if(adl[2] not in addrDict.keys()):
            addrDict[adl[2]] = typeDictValue
            typeDictValue += 1
        if(adl[1] not in addrDict.keys()):
            addrDict[adl[1]] = NameDictValue
            NameDictValue += 1
        if(adl[0] not in addrDict.keys()):
            addrDict[adl[0]] = adl[0]
    return addrDict

def addr2Vector(data, addrDict):
    # data = pd.read_csv('../input/train.csv')
    # addrDict = buildAddrDict(data)
    senVec = np.zeros(shape=(data.shape[0],3))

    for it,item in enumerate(data['Address'].values):
        item = item.strip(" /")
        if(' / ' in item):
            item = item.split(' / ')[0]
        tempV = addr2list(item)
        for i,t in enumerate(tempV):
            if(t in addrDict.keys()):
                tempV[i] = addrDict[t]
            else:
                tempV[i] = -1
            # print(tempV)
            # print(iitt)
        senVec[it] = tempV 
    # print("senVec vec len =", senVec.shape)
    # print(senVec)
    addressPD = pd.DataFrame(data=senVec,columns=['Block','RegionName','RegionType'])
    
    # print("yuan shi shu ju :data")
    # print(data.shape)
    # print(data)

    data = pd.concat([data,addressPD], axis=1, ignore_index=False)
    # print(data)
    return data

# # address to vec using doc2vec method
def addressToVector(data):
    model = Doc2Vec.load("./Doc_Word2vec-master/d2v200unique.model")
    senVec = np.zeros(shape=(data.shape[0],200))
    print("data shape:", data.shape)
    # print("address shape:", len(data['Address'].values))
    # print("Descript shape:", len(data['Descript'].values))
    # print("PdDistrict shape:", len(data['PdDistrict'].values))
    for it,item in enumerate(data['Address'].values):
        item = word_tokenize(item.lower())
        tempV = model.infer_vector(item)
        # print(it,tempV)
        senVec[it] = tempV 
    print("senVec vec len =", senVec.shape)
    print(senVec)
    addressPD = pd.DataFrame(data=senVec)
    
    print("yuan shi shu ju :data")
    print(data.shape)
    print(data)

    data = pd.concat([data,addressPD], axis=1, ignore_index=False)
    return data
    
    
    
def feature_engineering(data):
    data['Date'] = pd.to_datetime(data['Dates'].dt.date)
    data['n_days'] = (
        data['Date'] - data['Date'].min()).apply(lambda x: x.days)
    data['WeekOfYear'] = data['Dates'].dt.week
    data['Day'] = data['Dates'].dt.day
    data['DayOfWeek'] = data['Dates'].dt.weekday
    data['Month'] = data['Dates'].dt.month
    data['Year'] = data['Dates'].dt.year
    data['Hour'] = data['Dates'].dt.hour
    data['Minute'] = data['Dates'].dt.minute
    # data['Block'] = data['Address'].str.contains('block', case=False)
    
    # this will decrease the performance of the model
    # # use doc2vec method to convert address to 200 dim vector
    # data = addressToVector(data)
    
    data.drop(columns=['Dates', 'Date', 'Address'], inplace=True)
    # print("he bing: :data")
    # print(data.shape)
    # print(data)
    
    return data


def main():
    
    # Loading the data  nrows=100,
    train = pd.read_csv('./input/train.csv', parse_dates=['Dates'])
    test = pd.read_csv('./input/test.csv', parse_dates=['Dates'], index_col='Id')


    #### use pdDV and pdRV will Reduce performance of the model
    # # create vector for descript
    # pdDV = {}
    # pdd = train.groupby(['PdDistrict','Descript'],as_index=True)['Category'].size().reset_index(name='Size')

    # for row in pdd[{'PdDistrict','Size'}].itertuples():
    #     if(getattr(row, 'PdDistrict') in pdDV.keys()):
    #         pdDV[getattr(row, 'PdDistrict')].append(getattr(row, 'Size'))
    #     else:
    #         pdDV[getattr(row, 'PdDistrict')] = [getattr(row, 'Size')]

    # for pdname in train['PdDistrict'].unique():
    #     # print(pdname,end=":")
    #     molOfV = np.linalg.norm(pdDV[pdname])
    #     pdDV[pdname] = molOfV
    #     # print(molOfV)
    #     # print(sigmoid(molOfV))
    # print(pdDV)

    # # create vector for resolution
    # pdRV = {}
    # pdr = train.groupby(['PdDistrict','Resolution'],as_index=True)['Category'].size().reset_index(name='Size')
    # # for item in np.array(pdr[{'PdDistrict','Size'}]).tolist():
    # #     if(item[1] in pdRV.keys()):
    # #         pdRV[item[1]].append(item[0])
    # #     else:
    # #         pdRV[item[1]] = [item[0]]

    # for row in pdr[{'PdDistrict','Size'}].itertuples():
    #     if(getattr(row, 'PdDistrict') in pdRV.keys()):
    #         pdRV[getattr(row, 'PdDistrict')].append(getattr(row, 'Size'))
    #     else:
    #         pdRV[getattr(row, 'PdDistrict')] = [getattr(row, 'Size')]

    # for pdname in train['PdDistrict'].unique():
    #     # print(pdname,end=":")
    #     molOfV = np.linalg.norm(pdRV[pdname])
    #     pdRV[pdname] = molOfV
    #     # print(molOfV)
    #     # print(sigmoid(molOfV))
    # print(pdRV)

    # train['pdDV'] = train.apply(lambda x: pdDV[x.PdDistrict], axis=1)
    # train['pdRV'] = train.apply(lambda x: pdRV[x.PdDistrict], axis=1)

    # test['pdDV'] = test.apply(lambda x: pdDV[x.PdDistrict], axis=1)
    # test['pdRV'] = test.apply(lambda x: pdRV[x.PdDistrict], axis=1)

    # print(train.head)
    # print(test.head)

    le2 = LabelEncoder()
    # trainCpy = train.copy()
    y = le2.fit_transform(train['Category'])

    imp = SimpleImputer(strategy='mean')

    for district in train['PdDistrict'].unique():
        train.loc[train['PdDistrict'] == district, ['X', 'Y']] = imp.fit_transform(
            train.loc[train['PdDistrict'] == district, ['X', 'Y']])
        test.loc[test['PdDistrict'] == district, ['X', 'Y']] = imp.transform(
            test.loc[test['PdDistrict'] == district, ['X', 'Y']])
    train_data = lgb.Dataset(
        train, label=y, categorical_feature=['PdDistrict'], free_raw_data=False)
    
    # # simply use the address to dict vector will increase the performance of the model
    # # add address to vector using dict
    # train, dataDict = parameterToVector(train, 'Address', ' ',dataDict={},dimCount=0)
    # test, dataDict = parameterToVector(test, 'Address', ' ', dataDict, 10)
    
    addrDict = buildAddrDict(train)
    train = addr2Vector(train,addrDict)
    test = addr2Vector(test,addrDict)
    

    # Feature Engineering
    train = feature_engineering(train)
    train.drop(columns=['Descript', 'Resolution'], inplace=True)
    test = feature_engineering(test)

    #### clean "y=90" data will decrease the performance of the model
    # # Data cleaning
    # train.drop_duplicates(inplace=True)
    # train.replace({'X': -120.5, 'Y': 90.0}, np.NaN, inplace=True)
    # test.replace({'X': -120.5, 'Y': 90.0}, np.NaN, inplace=True)
    
    # Encoding the Categorical Variables
    le1 = LabelEncoder()
    train['PdDistrict'] = le1.fit_transform(train['PdDistrict'])
    test['PdDistrict'] = le1.transform(test['PdDistrict'])

    le2 = LabelEncoder()
    X = train.drop(columns=['Category'])
    y = le2.fit_transform(train['Category'])


    ## just get some figures ,not essential
    # 计算各个参数直接的相关性
    # pearson：Pearson相关系数来衡量两个数据集合是否在一条线上面，即针对线性数据的相关系数计算，针对非线性数据便会有误差。
    # kendall：用于反映分类变量相关性的指标，即针对无序序列的相关系数，非正态分布的数据
    # spearman：非线性的，非正太分析的数据的相关系数
    print(train.head())
    cor = test.corr()
    print(cor)

    # Creating the model
    train_data = lgb.Dataset(X, label=y, categorical_feature=['PdDistrict'])
    # train_data = lgb.Dataset(X, label=y)

    params = {'boosting': 'gbdt',
            'objective': 'multiclass',
            'num_class': 39,
            'max_delta_step': 0.9,
            'min_data_in_leaf': 21,
            'learning_rate': 0.4,
            'max_bin': 465,
            'num_leaves': 41
            }

    bst = lgb.train(params, train_data, 100) 
    t = time.strftime("%Y-%m-%d_%H%M%S", time.localtime())
    bst.save_model("./lgbm_" + str(t) + ".model")

    predictions = bst.predict(test)

    # Submitting the results
    submission = pd.DataFrame(
        predictions,
        columns=le2.inverse_transform(np.linspace(0, 38, 39, dtype='int16')),
        index=test.index)
    t = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    submission.to_csv('LGBM_' + str(t) +'.csv', index_label='Id')

if __name__ == "__main__":
    tStart = time.time()
    main()
    tEnd = time.time()
    print("time spend:" + str(tEnd - tStart))
