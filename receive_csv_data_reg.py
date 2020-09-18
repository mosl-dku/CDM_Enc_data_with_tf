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

sample_csv = 'age,sex_M,HEIGHT,WEIGHT,DATE,L3008,L3062,L3061,L3068,\
75,0,154.5,47.0,17/02/13,112,66,88,36,\
54,0,158.8,56.6,17/02/08,191,91,132,83,\
54,0,160.0,66.5,17/01/13,266,57,134,181,\
64,1,164.7,66.75,17/01/13,183,48,135,111,\
58,1,172.3,66.6,17/01/13,112,42,97,64,\
54,0,157.8,55.6,17/01/13,175,56,141,94,\
56,1,166.2,60.7,17/01/13,202,60,186,106,\
69,1,165.7,73.3,17/01/13,216,59,91,140,\
68,1,172.8,62.5,17/01/13,141,52,48,82,\
61,1,160.2,66.7,17/01/13,147,53,81,84,\
53,1,164.8,63.2,17/01/13,214,64,112,137,\
47,0,164.8,53.1,17/01/13,129,57,71,60,\
73,1,162.7,62.15,17/01/13,176,42,138,117,\
53,0,161.5,60.8,17/01/13,256,74,138,155,\
47,1,165.7,78.8,17/01/13,256,60,127,181,\
58,0,160.9,65.7,17/01/13,176,57,103,98,\
80,1,168.2,75.2,17/01/13,152,32,143,91,\
59,1,165.5,62.7,17/01/13,232,53,192,150,\
48,1,174.4,90.0,17/05/10,190,38,234,119,\
60,1,164.2,79.5,17/04/07,179,32,203,120,\
49,0,162.9,61.8,17/01/13,181,67,74,95,\
53,0,157.3,53.4,17/03/24,289,92,110,186,\
60,0,151.9,63.95,17/07/14,175,60,330,75,\
65,0,152.6,59.9,17/04/06,155,43,271,71,\
81,1,169.8,62.9,17/05/12,116,36,78,67,\
73,1,175.3,73.2,17/05/17,155,34,92,101,\
57,0,156.4,55.8,17/01/13,179,45,107,107,\
51,0,151.2,55.32,16/11/25,190,48,72,131,\
47,1,167.9,79.5,17/02/03,172,55,174,97,\
71,0,163.2,51.7,17/06/30,226,78,81,133,\
49,0,156.3,66.6,17/04/03,245,38,180,157,\
73,0,155.3,55.6,17/02/17,192,68,145,109,\
74,0,157.5,63.2,17/06/23,176,63,83,100,\
51,1,176.0,70.1,17/02/10,134,41,115,73,\
35,1,173.0,52.7,17/04/04,149,57,83,77,\
62,0,148.0,45.0,17/04/07,136,46,154,64,\
69,0,149.4,52.8,17/05/18,135,59,64,65,\
66,1,159.0,65.0,17/05/12,120,48,94,63,\
78,0,142.6,59.4,17/03/07,116,42,131,52,\
75,1,162.0,57.4,17/01/25,150,41,90,84,\
33,0,156.2,45.2,17/03/10,185,116,74,57,\
61,1,169.6,74.5,17/06/02,126,58,74,55,\
69,0,155.5,54.5,17/05/04,190,65,71,108,\
43,1,167.0,72.0,17/01/13,231,60,248,139,\
75,1,164.4,69.05,17/03/09,140,85,79,44,\
75,1,159.0,65.4,17/04/07,169,46,142,99,\
46,1,180.0,96.7,17/03/17,257,51,315,153,\
68,1,170.6,78.65,17/03/10,126,52,102,54,\
57,0,151.4,52.25,17/07/06,193,54,114,121,\
65,0,151.0,58.75,17/04/04,234,88,46,138,\
87,0,150.9,59.9,17/05/12,174,70,156,82,\
75,0,152.1,46.2,17/07/03,224,96,98,103,\
40,0,163.0,64.5,17/01/13,185,50,138,115,\
84,1,167.0,82.1,17/05/16,150,32,176,81,\
80,0,151.0,92.0,17/07/07,143,58,157,58,\
77,1,170.9,68.1,17/04/07,165,61,74,92,\
68,1,164.0,65.75,17/04/14,167,50,136,90,\
47,1,175.6,91.25,17/07/07,253,45,389,102,\
87,1,164.8,71.0,17/01/13,149,56,143,67,\
56,0,160.9,61.8,17/06/11,309,64,136,199,\
65,1,169.0,72.5,16/11/25,181,53,171,111,\
26,1,175.0,77.4,17/03/03,150,39,118,98,\
75,0,147.0,53.0,17/01/16,199,40,383,78,\
58,1,168.0,64.9,17/05/19,144,49,94,77,\
65,1,171.0,76.2,17/07/28,138,47,83,76,\
61,0,156.0,54.0,17/01/13,162,61,130,78,\
42,1,175.0,78.8,17/07/28,136,37,257,60'


x_train,x_test,y_train,y_test,test_x,test_y = make_df_preprocess(sample_csv)

n_epochs = 1000
learning_rate = 0.001

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
