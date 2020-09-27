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
sys.path.insert(1,'/home/mosl/CDM_sample_data/tf_grpc')
import decryptor_pb2_grpc
import decryptor_pb2
from client import sendCiphertext
import time
warnings.filterwarnings('ignore')


'''
def run():
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = helloworld_pb2_grpc.GreeterStub(channel)
        response = stub.GiveCSV(helloworld_pb2.HelloRequest())
        print("csv data is : " + response.message)
        return response.message
'''

def make_df_preprocess(sample_csv):
 # print("start preprocess\n")
  sample = sample_csv.split(',')

  index_li = []
  data = []
  tmp_li = []

  for x in sample:
    if any(tmp.isalpha() for tmp in x):
      index_li.append(x)
      index_len = len(index_li)
    else:
      break

  for x in range(len(index_li),len(sample),len(index_li)):
      data.extend([sample[x:x+len(index_li)]])

  train = pd.DataFrame(data)
  train.columns = index_li

  	
  train.drop(['DATE'],axis=1,inplace=True)
  ch_list = ['age','sex_M', 'HEIGHT', 'WEIGHT', 'L3008', 'L3062', 'L3061','L3068']
  for x in ch_list:
    train[x] = pd.to_numeric(train[x],downcast='float')
  
  train = train[(train['L3008'] < 650)]
  train = train[(train['HEIGHT'] < 250)]

  train = train[(train['WEIGHT'] < 200)]
  numeric_feats = train.dtypes[train.dtypes != "object"].index
  test_sample = train[3000:]
  train = train[:3000]
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
  #print("end preprocess")
  return x_train,x_test,y_train,y_test,test_x,test_y



###################################################################


logging.basicConfig()
#f = open("/home/mosl/CDM_sample_data/divide_dataset_2/data0.txt.enc",'rb')
#dat = f.read()
#f.close
theta = tf.Variable(tf.random_uniform([7,1],-1.0,1.0),name="theta")
init = tf.global_variables_initializer()

n_epochs = 1000
learning_rate = 0.001
#sample_csv = sendCiphertext(dat,'/home/mosl/CDM_sample_data/divide_dataset_2/data0.txt.enc.key')
#sample_csv = sample_csv.replace("\n","")
#x_train,x_test,y_train,y_test,test_x,test_y = make_df_preprocess(sample_csv)

ti = time.time()
with tf.Session() as sess:
  sess.run(init)

  for epoch in range(n_epochs):
    if epoch==0:
      f = open("/home/mosl/CDM_sample_data/divide_dataset/data0.txt.enc",'rb')
      dat = f.read()
      f.close()
      sample_csv = sendCiphertext(dat,'/home/mosl/CDM_sample_data/divide_dataset/data0.txt.enc.key')
      sample_csv = sample_csv.replace("\n","")
      x_train,x_test,y_train,y_test,test_x,test_y = make_df_preprocess(sample_csv)
     


    X = tf.constant(x_train,dtype=tf.float32,name="X")
    y = tf.constant(y_train,dtype=tf.float32,name="y")   
    y_pred = tf.matmul(X,theta,name="predictions")
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error),name="mse")
    gradients = tf.gradients(mse,[theta])[0]
    training_op = tf.assign(theta,theta-learning_rate*gradients)    
    sess.run(training_op)
    if epoch%100==0:
      print("Epoch",epoch,"mse = ",mse.eval())
      f_num = (epoch%1200)//100
      #print(f_num)
      f_name = '/home/mosl/CDM_sample_data/divide_dataset/data'+str(f_num)+'.txt.enc'
      k_name = '/home/mosl/CDM_sample_data/divide_dataset/data'+str(f_num)+'.txt.enc.key'
      f = open(f_name,'rb')
      dat = f.read()
      f.close()
      sample_csv = sendCiphertext(dat,k_name)
      sample_csv = sample_csv.replace("\n","")	
      x_train,x_test,y_train,y_test,test_x,test_y = make_df_preprocess(sample_csv)

  best_theta = theta.eval()

print("time is :",ti - time.time())


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
