#!/usr/bin/env python3

'''
ref: https://mjdeeplearning.tistory.com/5
'''


import pandas as pd
import statsmodels.api as sm
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import numpy as np

from pprint import pprint

wine_data = pd.read_csv('../data/winequality-white.csv', delimiter=';', dtype=float)
wine_data.head(10)

pprint(wine_data)

x_data = wine_data.iloc[:,0:-1]
y_data = wine_data.iloc[:,-1]

# Score 값이 7보다 작으면 0,  7보다 크거나 같으면 1로 값 변경.
y_data = np.array([1 if i >= 7 else 0 for i in y_data])
x_data.head(5)

print('-- x_data --')
pprint(x_data)
print('-- y_data --')
pprint(y_data)

# 트레인, 테스트 데이터 나누기.
train_x, test_x, train_y, test_y = sklearn.model_selection.train_test_split(x_data, y_data, test_size = 0.3,random_state=42)

# logistic regression model
log_reg = LogisticRegression()
log_reg.fit(train_x, train_y)

# evaluation
y_pred = log_reg.predict(test_x)
print("Train Data:",log_reg.score(train_x,train_y))
print("Test Data",sum(y_pred == test_y) / len(test_y))

from sklearn.metrics import classification_report
y_true, y_pred = test_y, log_reg.predict(test_x)
print(classification_report(y_true, y_pred))

logit = sm.Logit(train_y,train_x).fit()
logit.summary()
print(np.exp(logit.params))

params = logit.params
conf = logit.conf_int()
conf['OR'] = params
conf.columns = ['2.5%', '97.5%', 'OR']
print (np.exp(conf))
