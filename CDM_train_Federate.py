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
warnings.filterwarnings('ignore')


l_train = pd.read_csv('dataset_age_G_H_W_D.csv')
l_train.drop(['DATE'],axis=1,inplace=True)
l_train =l_train[(l_train['L3008'] < 650)]
l_train = l_train[(l_train['HEIGHT'] < 250)]

l_train = l_train[(l_train['WEIGHT'] < 200)]
numeric_feats = l_train.dtypes[l_train.dtypes != "object"].index
test_sample = l_train[400000:]
train = l_train[:100000]

log_list = ['age', 'HEIGHT', 'WEIGHT', 'L3008', 'L3062', 'L3061']
for sample_idx in log_list:
  train[sample_idx] = train[sample_idx]/100

train_y = train['L3068'].to_frame()
train.drop(['L3068'],axis=1,inplace=True)

x = np.array(train)
y = np.array(train_y)
x[np.isinf(x)]=0
y[np.isinf(y)]=0


x_train, x_test, y_train, y_test = train_test_split(x, y,test_size = .3, random_state=0)

test_y = test_sample['L3068'].to_frame()
test_sample.drop(['L3068'],axis=1,inplace=True)
test_x = test_sample

for sample_idx in log_list:
  test_x[sample_idx] = test_x[sample_idx]/100

n_epochs = 200000
learning_rate = 0.0001

X = tf.constant(x_train,dtype=tf.float32,name="X")
y = tf.constant(y_train,dtype=tf.float32,name="y")
theta = tf.Variable(tf.random_uniform([7,1],-1.0,1.0),name="theta")
y_pred = tf.matmul(X,theta,name="predictions")



error = y_pred - y
mse = tf.reduce_mean(tf.square(error),name="mse")
gradients = tf.gradients(mse,[theta])[0]
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9)
training_op = optimizer.minimize(mse)


init = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(init)

  for epoch in range(n_epochs):
    if epoch%100==0:
      print("Epoch",epoch,"mse = ",mse.eval())
    sess.run(training_op)

  best_theta1 = theta.eval()
#########################################################
train = l_train[100000:200000]

log_list = ['age', 'HEIGHT', 'WEIGHT', 'L3008', 'L3062', 'L3061']
for sample_idx in log_list:
  train[sample_idx] = train[sample_idx]/100

train_y = train['L3068'].to_frame()
train.drop(['L3068'],axis=1,inplace=True)

x = np.array(train)
y = np.array(train_y)
x[np.isinf(x)]=0
y[np.isinf(y)]=0


x_train, x_test, y_train, y_test = train_test_split(x, y,test_size = .3, random_state=0)

#test_y = test_sample['L3068'].to_frame()
#test_sample.drop(['L3068'],axis=1,inplace=True)
#test_x = test_sample

#for sample_idx in log_list:
#  test_x[sample_idx] = test_x[sample_idx]/100

n_epochs = 200000
learning_rate = 0.0001

X = tf.constant(x_train,dtype=tf.float32,name="X")
y = tf.constant(y_train,dtype=tf.float32,name="y")
theta = tf.Variable(tf.random_uniform([7,1],-1.0,1.0),name="theta")
y_pred = tf.matmul(X,theta,name="predictions")



error = y_pred - y
mse = tf.reduce_mean(tf.square(error),name="mse")
gradients = tf.gradients(mse,[theta])[0]
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9)
training_op = optimizer.minimize(mse)


init = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(init)

  for epoch in range(n_epochs):
    if epoch%100==0:
      print("Epoch",epoch,"mse = ",mse.eval())
    sess.run(training_op)

  best_theta2 = theta.eval()
###################################
train = l_train[200000:300000]

log_list = ['age', 'HEIGHT', 'WEIGHT', 'L3008', 'L3062', 'L3061']
for sample_idx in log_list:
  train[sample_idx] = train[sample_idx]/100

train_y = train['L3068'].to_frame()
train.drop(['L3068'],axis=1,inplace=True)

x = np.array(train)
y = np.array(train_y)
x[np.isinf(x)]=0
y[np.isinf(y)]=0


x_train, x_test, y_train, y_test = train_test_split(x, y,test_size = .3, random_state=0)

#test_y = test_sample['L3068'].to_frame()
#test_sample.drop(['L3068'],axis=1,inplace=True)
#test_x = test_sample

#for sample_idx in log_list:
#  test_x[sample_idx] = test_x[sample_idx]/100

n_epochs = 200000
learning_rate = 0.0001

X = tf.constant(x_train,dtype=tf.float32,name="X")
y = tf.constant(y_train,dtype=tf.float32,name="y")
theta = tf.Variable(tf.random_uniform([7,1],-1.0,1.0),name="theta")
y_pred = tf.matmul(X,theta,name="predictions")



error = y_pred - y
mse = tf.reduce_mean(tf.square(error),name="mse")
gradients = tf.gradients(mse,[theta])[0]
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9)
training_op = optimizer.minimize(mse)


init = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(init)

  for epoch in range(n_epochs):
    if epoch%100==0:
      print("Epoch",epoch,"mse = ",mse.eval())
    sess.run(training_op)

  best_theta3 = theta.eval()
###################################
train = l_train[300000:400000]

log_list = ['age', 'HEIGHT', 'WEIGHT', 'L3008', 'L3062', 'L3061']
for sample_idx in log_list:
  train[sample_idx] = train[sample_idx]/100

train_y = train['L3068'].to_frame()
train.drop(['L3068'],axis=1,inplace=True)

x = np.array(train)
y = np.array(train_y)
x[np.isinf(x)]=0
y[np.isinf(y)]=0


x_train, x_test, y_train, y_test = train_test_split(x, y,test_size = .3, random_state=0)

#test_y = test_sample['L3068'].to_frame()
#test_sample.drop(['L3068'],axis=1,inplace=True)
#test_x = test_sample

#for sample_idx in log_list:
#  test_x[sample_idx] = test_x[sample_idx]/100

n_epochs = 200000
learning_rate = 0.0001

X = tf.constant(x_train,dtype=tf.float32,name="X")
y = tf.constant(y_train,dtype=tf.float32,name="y")
theta = tf.Variable(tf.random_uniform([7,1],-1.0,1.0),name="theta")
y_pred = tf.matmul(X,theta,name="predictions")



error = y_pred - y
mse = tf.reduce_mean(tf.square(error),name="mse")
gradients = tf.gradients(mse,[theta])[0]
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9)
training_op = optimizer.minimize(mse)


init = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(init)

  for epoch in range(n_epochs):
    if epoch%100==0:
      print("Epoch",epoch,"mse = ",mse.eval())
    sess.run(training_op)

  best_theta4 = theta.eval()
###################################





##############################
cz1 = np.transpose(best_theta1)
cz2 = np.transpose(best_theta2)
cz3 = np.transpose(best_theta3)
cz4 = np.transpose(best_theta4)
te = np.array(test_x)
ans_list1 = []
ans_list2 = []
ans_list3 = []
ans_list4 = []

final_ans = []

for se in range(100):
  res = np.matmul(cz1,te[se])
  ans_list1.append(res)
  res = np.matmul(cz2,te[se])
  ans_list2.append(res)
  res = np.matmul(cz3,te[se])
  ans_list3.append(res)
  res = np.matmul(cz4,te[se])
  ans_list4.append(res)

for se in range(100):
  res  = (ans_list1[se]+ans_list2[se]+ans_list3[se]+ans_list4[se])/4
  final_ans.append(res)


print(final_ans)
print("#################################\n\n\n\n")
print(test_y.head(100))

te_y = np.array(test_y)

diff_li = []

for se in range(100):
  res = abs(te_y[se]-final_ans[se])
  diff_li.append(res)

print("diff :",diff_li)
