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


train = pd.read_csv('dataset_age_G_H_W_D.csv')
train.drop(['DATE'],axis=1,inplace=True)
train = train[(train['L3008'] < 650)]
train = train[(train['HEIGHT'] < 250)]

train = train[(train['WEIGHT'] < 200)]
numeric_feats = train.dtypes[train.dtypes != "object"].index
test_sample = train[400000:]
train = train[:400000]

log_list = ['age', 'HEIGHT', 'WEIGHT', 'L3008', 'L3062', 'L3061']
for sample_idx in log_list:
  train[sample_idx] = np.log(1+train[sample_idx])

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


learning_rate = 0.1     
training_epochs = 3   
display_step = 2   

X = tf.placeholder(tf.float32)  
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_uniform([1, 7], -1.0, 1.0), name="weight")
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name="bias")

pred = tf.add(tf.multiply(X, W), b)   # pred = X * W + b

cost = tf.reduce_mean(tf.square(pred - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


with tf.Session() as sess:

    
    sess.run(tf.global_variables_initializer())

    num=0
    for epoch in range(training_epochs):
        for (x, y) in zip(x_train, y_train):
          num+=1
          if num%1000==0:
            print(num)

          sess.run(optimizer, feed_dict={X: x, Y: y}) 

       
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: x_train, Y:y_train})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "W=", sess.run(W), "b=", sess.run(b))


    print("최적화가 완료되었습니다.")
    training_cost = sess.run(cost, feed_dict={X: x_train, Y: y_train})
    print("훈련이 끝난 후 비용과 모델 파라미터입니다.  cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')
    re_W = sess.run(W)
    re_b = sess.run(b)

print(re_W,'\n')
print(re_b)
 
