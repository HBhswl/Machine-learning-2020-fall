# Machine-learning-2020-fall

This repo is the implementation of our team's algorithm for the competition [San Francisco Crime Classification](https://www.kaggle.com/c/sf-crime)

## Getting started

### Requirements

- Python version == 3.7.9
- pandas == 1.0.5
- scikit-learn == 0.23.2
- lightgbm == 2.3.0
- gensim == 3.8.3
- xgboost == 1.3.1
- catboost == 0.24.4

## Usage

You should download the data (train.csv, test.csv) from [Kaggle](https://www.kaggle.com/c/sf-crime/data), and put them in the current directory.

### Generate Features

You can use the following command to gernrate the features.

```python
$ python demo.py
```

### Train models and get the outputs on test

We have provided six different models. You can choose one of these following commands to get your outputs.

```python
$ python model/stack_naivebayes.py --output "yourpath"
$ python model/stack_knn.py --output "yourpath"
$ python model/stack_rfr.py --output "yourpath"
$ python model/stack_lightgbm.py --output "yourpath"
$ python model/stack_xgboost.py --output "yourpath"
$ python model/stack_catboost.py --output "yourpath"
```



