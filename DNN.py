import pandas as pd 
import numpy as np 
import pandas as pd
import math
import sklearn.model_selection as ms
import sklearn.metrics as sklm
from sklearn import linear_model
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import warnings
from scipy.stats import skew
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
import math
import sklearn.model_selection as ms
import sklearn.metrics as sklm
from tensorflow.python.keras.callbacks import TensorBoard
from time import time
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

test_y = test_sample['L3068'].to_frame()
test_sample.drop(['L3068'],axis=1,inplace=True)
test_x = test_sample



model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu, input_shape=[x_train.shape[1]]),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(64, activation=tf.nn.leaky_relu),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(16, activation=tf.nn.leaky_relu),
    tf.keras.layers.Dropout(0.1),
tf.keras.layers.Dense(2, activation=tf.nn.leaky_relu),
    tf.keras.layers.Dropout(0.1),   
 tf.keras.layers.Dense(1)
])

optimizer = tf.keras.optimizers.Adam(0.01)

tensorboard = TensorBoard(log_dir='/home/mosl/CDM_sample_data/log_file/{}'.format(time()))

model.compile(loss='mean_squared_logarithmic_error',
              optimizer=optimizer,
              metrics=['mean_squared_logarithmic_error'])

print(model.summary())

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_mean_squared_logarithmic_error', patience=10)

history = model.fit(x_train,
                   y_train,
                   batch_size=10000,
                   epochs=100,
                   callbacks=[tensorboard],
                   validation_data=(x_test, y_test))

mse, _ = model.evaluate(x_test, y_test)
rmse = np.sqrt(mse)
print('Root Mean Square Error on test set: {}'.format(round(rmse, 3)))


def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch  
  plt.figure(figsize=(12,8))
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_logarithmic_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_logarithmic_error'],
           label = 'Val Error')
  plt.ylim([0,.1])
  plt.legend()
  plt.show()

print(plot_history(history))

res = model.predict(test_x)
print(res)


print("#################################\n\n\n\n")
print(test_y.head(100))

te_y = np.array(test_y)

diff_li = []
su =0
for se in range(100):
  re = abs(te_y[se]-res[se])
  su+=re[0]
  diff_li.append(re)

print("diff :",diff_li)

print(su)
