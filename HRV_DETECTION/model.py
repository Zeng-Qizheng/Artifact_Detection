# _*_ coding: utf-8 _*_

"""
@ date:             2020-11-20
@ author:           jingxian
@ illustration:     model result
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, RocCurveDisplay, confusion_matrix
import xgboost as xgb
import sys
import logging
from imblearn.under_sampling import RandomUnderSampler


if os.path.exists("./model.log"):
    os.remove("./model.log")

logging.basicConfig(level=logging.DEBUG,filename="model.log", filemode='a',format="%(message)s")


def RandomForest(train_data, train_label, test_data, test_label, test_index):
    model = RandomForestClassifier(
        n_estimators=200,
        criterion='entropy',
        max_depth=4,
        random_state=1
    )
    model.fit(train_data, train_label)
    importances = model.feature_importances_
    logging.info("***********************" + "RandomForest" + "***************************")
    logging.info("DecisionTree feature importances :")
    for col, importance in zip(columns, importances):
        logging.info(col + ":" + str(importance))
    predict = model.predict(test_data)
    logging.info("RF_index:")
    logging.info(" ".join(str(x) for x in test_index))
    logging.info("RF_label")
    logging.info(" ".join(str(x) for x in test_label.to_numpy().reshape(-1)))
    logging.info("RF_predict:")
    logging.info(" ".join(str(x) for x in predict))
    logging.info("RF_ACC:" + str(accuracy_score(test_label, predict)))
    logging.info("RF_Recall:" + str(recall_score(test_label, predict, average="micro")))
    logging.info("RF_Precision:" + str(precision_score(test_label, predict, average="micro")))
    logging.info("RF_f1:" + str(f1_score(test_label, predict, average="micro")))
    logging.info("RF_confusion_matrix:")
    logging.info(confusion_matrix(test_label, predict))
    # print(predict[np.where(test_label != predict)])
    return np.where(test_label != predict), predict


def XGboost(train_data, train_label, test_data, test_label, test_index):
    model = xgb.XGBClassifier(
        max_depth=4,
        learning_rate=0.01,
        n_estimators=100,
        min_child_weight=5,
        max_delta_step=0,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0,
        reg_lambda=0.4,
        missing=None,
        eval_metric='auc',
        seed=1440,
    )
    model.fit(train_data, train_label)
    importances = model.feature_importances_
    logging.info("***********************" + "XGboost" + "***************************")
    logging.info("DecisionTree feature importances :")
    for col, importance in zip(columns, importances):
        logging.info(col + ":" + str(importance))
    predict = model.predict(test_data)
    logging.info("XG_index:")
    logging.info(" ".join(str(x) for x in test_index))
    logging.info("XG_label:")
    logging.info(" ".join( str(x) for x in test_label.to_numpy().reshape(-1)) )
    logging.info("XG_predict:")
    logging.info(" ".join(str(x) for x in predict))
    logging.info("xg_boost_ACC:" + str(accuracy_score(test_label, predict)))
    logging.info("xg_boost_Recall:" + str(recall_score(test_label, predict, average="micro")))
    logging.info("xg_boost_Precision:" + str(precision_score(test_label, predict, average="micro")))
    logging.info("xg_boost_f1:" + str(f1_score(test_label, predict, average="micro")))
    logging.info("xg_boost_confusion_matrix:")
    logging.info(confusion_matrix(test_label, predict))
    # print(predict[np.where(test_label != predict)])
    return np.where(test_label != predict), predict


df = pd.read_csv("./data_info.csv")
print( df.groupby('new_label').count() )
model_RandomUnderSample = RandomUnderSampler()          # 建立RandomUnderSampler模型对象
columns = df.columns[3:]
df_label3 = df[ df["new_label"] == 3 ]
df_label12 = df[ df["new_label"] != 3 ]

x = df_label12[columns]
y = df_label12["new_label"]
x_resampled, y_resampled = model_RandomUnderSample.fit_sample(x, y)
df_resampled = pd.concat([x_resampled, y_resampled], axis = 1)
print(df_resampled)
#print( x_resampled )
#print( y_resampled )


sys.exit()
columns = df.columns[3:]
features = df[columns]
label = df["new_label"]

sfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
# ss = StratifiedShuffleSplit(n_splits=1,test_size=0.3,random_state=42)
for train_index, test_index in sfold.split(features, label):
    X_train = features.iloc[train_index, :]
    y_train = label[train_index]
    X_test = features.iloc[test_index, :]
    y_test = label[test_index]
    # DecisionTree(X_train, y_train, X_test, y_test)
    err_index1, RF_predict = RandomForest(X_train, y_train, X_test, y_test,test_index)
    err_index2, XG_predict = XGboost(X_train, y_train, X_test, y_test,test_index)
    print(test_index[err_index1])
    print(test_index[err_index2])

