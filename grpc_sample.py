from __future__ import print_function
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
import logging
import grpc
import sys
sys.path.insert(1,'/home/mosl/grpc/examples/python/helloworld')
import helloworld_pb2
import helloworld_pb2_grpc

warnings.filterwarnings('ignore')



def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = helloworld_pb2_grpc.GreeterStub(channel)
        response = stub.GiveCSV(helloworld_pb2.HelloRequest())
        print("csv data is : " + response.message)
        return response.message


def make_df_preprocess(sample_csv):
  sample = sample_csv.split(',')

  index_li = []
  data = []
  tmp_li = []

  for x in sample:
    if any(tmp.isalpha() for tmp in x):
      index_li.append(x)
      index_len = len(index_li)
    else:
      if len(tmp_li) != len(index_li):
        tmp_li.append(x)
      if len(tmp_li) == len(index_li):
        data.append(tmp_li)
        tmp_li = []

  train = pd.DataFrame(columns=index_li)
  for x in data:
    train.loc[len(train)] = x

  	
  train.drop(['DATE'],axis=1,inplace=True)
  ch_list = ['age','sex_M', 'HEIGHT', 'WEIGHT', 'L3008', 'L3062', 'L3061','L3068']
  for x in ch_list:
    train[x] = pd.to_numeric(train[x],downcast='float')
  
  train = train[(train['L3008'] < 650)]
  train = train[(train['HEIGHT'] < 250)]

  train = train[(train['WEIGHT'] < 200)]
  numeric_feats = train.dtypes[train.dtypes != "object"].index
  test_sample = train[60:]
  train = train[:60]
  down_list = ['age', 'HEIGHT', 'WEIGHT', 'L3008', 'L3062', 'L3061']
  for sample_idx in down_list:
    train[sample_idx] = train[sample_idx]/100

  train_y = train['L3068'].to_frame()
  train.drop(['L3068'],axis=1,inplace=True)

  x = np.array(train)
  y = np.array(train_y)

  x_train, x_test, y_train, y_test = train_test_split(x, y,test_size = .3, random_state=0)

  test_y = test_sample['L3068'].to_frame()
  test_sample.drop(['L3068'],axis=1,inplace=True)
  test_x = test_sample

  for sample_idx in down_list:
    test_x[sample_idx] = test_x[sample_idx]/100
  
  return x_train,x_test,y_train,y_test,test_x,test_y


logging.basicConfig()
sample_csv = run()

x_train,x_test,y_train,y_test,test_x,test_y = make_df_preprocess(sample_csv)

n_epochs = 100000
learning_rate = 0.001

X = tf.constant(x_train,dtype=tf.float32,name="X")
y = tf.constant(y_train,dtype=tf.float32,name="y")
theta = tf.Variable(tf.random_uniform([7,1],-1.0,1.0),name="theta")
y_pred = tf.matmul(X,theta,name="predictions")



error = y_pred - y
mse = tf.reduce_mean(tf.square(error),name="mse")
gradients = tf.gradients(mse,[theta])[0]
#optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9)
#training_op = optimizer.minimize(mse)
training_op = tf.assign(theta,theta-learning_rate*gradients)

init = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(init)

  for epoch in range(n_epochs):
    if epoch%100==0:
      print("Epoch",epoch,"mse = ",mse.eval())
    sess.run(training_op)
  
  best_theta = theta.eval()


cz = np.transpose(best_theta)
te = np.array(test_x)
ans_list = []

for se in range(5):
  res = np.matmul(cz,te[se])
  ans_list.append(res)

print(ans_list)
print("#################################\n\n\n\n")
print(test_y.head(10))

te_y = np.array(test_y)

diff_li = []
su=0
for se in range(5):
  res = abs(te_y[se]-ans_list[se])
  su+=res[0]
  diff_li.append(res)

print("diff :",diff_li)
print(su)
