import pandas as pd 
import numpy as np 
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
import math
import sklearn.model_selection as ms
import sklearn.metrics as sklm
from sklearn import linear_model
from sklearn.metrics import make_scorer
from sklearn import svm
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import warnings
from scipy.stats import skew
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
import math
import sklearn.model_selection as ms
import sklearn.metrics as sklm
warnings.filterwarnings('ignore')


train = pd.read_csv('dataset_age_G_H_W_D.csv')
train.drop(['DATE'],axis=1,inplace=True)
train = train[(train['L3008'] < 650)]
train = train[(train['HEIGHT'] < 250)]
train = train[(train['WEIGHT'] < 200)]
numeric_feats = train.dtypes[train.dtypes != "object"].index
test_sample = train[400000:]
train = train[:400000]
numeric_feats = ['age', 'HEIGHT', 'WEIGHT', 'L3008', 'L3062', 'L3061', 'L3068']

train_y = train['L3068'].to_frame()
train.drop(['L3068'],axis=1,inplace=True)

x = np.array(train)
y = np.array(train_y)



x_train, x_test, y_train, y_test = train_test_split(x, y,test_size = .3, random_state=0)

lreg=LinearRegression()
lin_reg = lreg.fit(x_train,y_train)
lin_pred = lin_reg.predict(x_test)
print('rmse test = ' + str(math.sqrt(sklm.mean_squared_error(y_test, lin_pred))))
print(lreg.coef_)
print(lreg.intercept_)
