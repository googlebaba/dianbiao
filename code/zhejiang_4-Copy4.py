
# coding: utf-8

# ## 此版本将测试集中出现的情况删除，以及去除了故障月份和使用时长

# In[2]:

# 浙江省第四类故障预测


# In[3]:



import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
sns.set_style('whitegrid')
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage
from scipy.sparse import coo_matrix, bmat
#get_ipython().magic(u'matplotlib inline')


# # Data Explore

# In[4]:

data = pd.read_csv('/usr/local/hadoop/src/code/dianbiao/data/all_4.csv', dtype={0:object, 'ORG':object, 'SPEC_CODE':object, 'MANUFACTURER':object,
                                                  'FAULT_MONTH':int, 'INST_MONTH':object, 'FAULT_QUARTER':object,
                                                  'FAULT_TYPE': object,'SYNC_ORG_NO':object,'ORG_NO':object
                                                 }, encoding='utf-8', nrows=10000)

#preview the zhejiang_4 data
data.head()


# In[5]:

data.info()


# # feature preprocessing

# In[6]:

#delete QUIP_ID
#data.drop([data.columns[0]], axis=1, inplace=True)


# In[7]:

data.drop([data.columns[0]], axis=1, inplace=True)
#data.drop_duplicates(['FAULT_TYPE', 'ORG_NO', 'SPEC_CODE', 'COMM_MODE', 'MANUFACTURER', 'FAULT_MONTH',
 #                    'INST_MONTH', 'month'], inplace=True)


# In[8]:

data.info()
data['FAULT_TYPE'].value_counts()


# In[9]:

fig, axis0 = plt.subplots(1, 1)
sns.countplot(x='FAULT_TYPE', data=data, ax=axis0)


# 从故障类型柱状图可以看出故障类型数据不平衡，402-406较少，407-411较多

# ## SYNC_ORG_NO

# In[10]:

#ORG
print data['SYNC_ORG_NO'].describe()
#plot
def plot_fun(name_fea, name_fault, fontsize=None):

    fig, axis1 = plt.subplots(1, 1)
    sns.countplot(x=name_fea, data=data, ax = axis1)

    fig, axis2 = plt.subplots(1, 1)
    c = data[name_fea].value_counts()
    s = c.cumsum()/c.sum()
    axis2.plot(np.arange(s.shape[0])+1, s.values*100)
    axis2.set_title('precent of %s'%name_fea)

    fig, axis3 = plt.subplots(1, 1)
    sns.countplot(x=name_fea, hue=name_fault, data=data, ax=axis3)
    plt.legend(loc = 2)

    fig, axis4 = plt.subplots(1, 1)
    sns.countplot(x=name_fault, hue=name_fea, data=data, ax=axis4)
    plt.legend(loc = 2, fontsize=fontsize)

    #calculate similar score
    from scipy.cluster.hierarchy import dendrogram, linkage
    #clustermap

    fault_num1 = data.groupby([name_fault, name_fea])[data.columns[0]].count().unstack()

    ratio = fault_num1 / fault_num1.sum()

    g1 = sns.clustermap(ratio,
                        cmap=plt.get_cmap('RdBu'),
                        vmax=1,
                        vmin=-1,
                        linewidth=0,
                        figsize=(10, 10),
                        row_cluster=False,
                        col_cluster=False
                    )
    plt.title('fault ratio')

#plot
#plot_fun('SYNC_ORG_NO', 'FAULT_TYPE')
#One-hot encoder
def onehot_pre(name):
    global data
    le  = preprocessing.LabelEncoder()
    le.fit(data[name])
    cat_name = list(le.classes_)
    data[name] = le.transform(data[name])
#    enc = preprocessing.OneHotEncoder()
#    enc.fit(np.array(data[name]).reshape(()))
    return cat_name
name_sync_org_no = onehot_pre('SYNC_ORG_NO')


# In[11]:


#ORG
data['ORG_NO'].describe()

#plot_fun('ORG_NO', 'FAULT_TYPE')

ORG_freq = data['ORG_NO'].value_counts().index[data['ORG_NO'].value_counts().values<100]
data['ORG_NO'] = data['ORG_NO'].replace(ORG_freq.values, 0)

name_org_no = onehot_pre('ORG_NO')


# ## ORG故障类型统计
# - 各个地区的故障数量不同，前8个到95%
# - 从ORG与FAULT_TYPE统计图可以看出，不同地区的故障类型分布有所不同，所以认为ORG对于FAULT_TYPE类型的识别是有用的。
# - 故障类型分布图显示了每个地区的故障类型占比
# - 有几个地区故障类型数据较少[33101,33407,33411]，对于故障类型识别用处不大，删除
# - 对属性做了二元变换处理

# ## SPEC_CODE

# In[12]:

#SPEC_CODE
data['SPEC_CODE'].describe()


# In[13]:

data['SPEC_CODE'].value_counts()


# In[14]:

spec_freq = data['SPEC_CODE'].value_counts().index[data['SPEC_CODE'].value_counts().values<500]
#spec_mapping = {label:idx for label,idx in zip(spec_freq, np.zeros(len(spec_freq)))}
print spec_freq.values


# In[15]:

data['SPEC_CODE'].value_counts()
data['SPEC_CODE'] = data['SPEC_CODE'].replace(spec_freq.values, 0)
print data['SPEC_CODE'].value_counts()


# In[16]:

#plot
#plot_fun('SPEC_CODE', 'FAULT_TYPE')
name_spec_code = onehot_pre('SPEC_CODE')


# ## SPEC_CODE故障类型统计
# - SPEC_CODE故障类型同样呈现分布不均匀状态
# - 前两类设备类型数据达到98%
# - 每种故障类型的SPEC_CODE基本相似
# - 故障类型分布图显示了每种SPEC_CODE故障类型占比
# - 删除极少出现的SPEC_CODE故障类型
# - 对属性进行二元变换

# # MANUFACTURER

# In[17]:

data['MANUFACTURER'].value_counts()


# In[18]:

spec_freq = data['MANUFACTURER'].value_counts().index[data['MANUFACTURER'].value_counts().values<500]
data['MANUFACTURER'] = data['MANUFACTURER'].replace(spec_freq.values, 0)
print len(data['MANUFACTURER'].value_counts())


# In[19]:

#plot
#plot_fun('MANUFACTURER', 'FAULT_TYPE', fontsize=1)

#cluster encoding
from scipy.cluster.hierarchy import fclusterdata
def cluster_encoding(name):
    global data
    fault_num = data.groupby(['FAULT_TYPE', name])[data.columns[0]].count().unstack()
    MAN_ratio = fault_num / fault_num.sum()
    MAN_ratio_T = MAN_ratio.T

    clusters = fclusterdata(np.array(MAN_ratio_T), 1)
    clusters_mapping = {label:idx for label,idx in zip(MAN_ratio.columns, clusters)}
    data[name] = data[name].map(clusters_mapping)

#对供应商进行相似度聚类
cluster_encoding('MANUFACTURER')
name_man = onehot_pre('MANUFACTURER')


# ## MANUFACTURER故障类型统计
# - MANUFACTURER故障类型同样呈现分布不均匀状态,浙江省一共有80家供应商，电表数前30家占90%
# - 前两类故障类型数据达到98%
# - 每种故障类型的供应商分布不同
# - 故障类型分布图显示了每种供应商故障类型占比，应用分层聚类方法将具有相似故障类型分布的供应商进行合并
# - 对属性进行二元变换

# # MONTH

# In[20]:

'''
# use month distribution
c1 = data.groupby(['month']).size()
c1.plot(kind='bar', figsize=(12, 6))

c2 = data.groupby(['month', 'FAULT_TYPE']).size().unstack().reindex(index=np.arange(data.month.min(), data.month.max()+1)).fillna(0)
c2.plot(kind='bar', figsize=(12, 12), subplots=True)

c3 = data.groupby(['month', 'SYNC_ORG_NO']).size().unstack().reindex(index=np.arange(data.month.min(), data.month.max()+1)).fillna(0)
c3.plot(kind='bar', figsize=(12, 12), subplots=True)
'''
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
data['month'] = min_max_scaler.fit_transform(data['month'])


# ## 使用寿命-故障类型统计
# - 对浙江省故障电表使用寿命进行了统计，可看出其分布基本为正态分布，符合客观规律。
# - 使用寿命-故障类型图显示了每种故障类型的使用寿命分布情况，基本为正态分布，但是其分布参数有所不同，可以用来作为分类特征。
# - 使用寿命-供电所分布图，不同供电所的使用寿命分布有区别，可以得出供电所对电表使用寿命有影响。

# ## FAULT_MONTH

# In[21]:

data['FAULT_MONTH'] = pd.Categorical(data['FAULT_MONTH'], ordered=True)

#m1 = data.groupby(['FAULT_MONTH', 'FAULT_TYPE']).size().unstack().reindex(index=np.arange(data.FAULT_MONTH.min(), data.FAULT_MONTH.max()+1)).fillna(0)
#m1.plot(kind='bar', figsize=(12, 12), subplots=True)
#plot_fun('FAULT_MONTH', 'FAULT_TYPE', fontsize=1)
'''
fault_num4 = data.groupby(['FAULT_TYPE', 'FAULT_MONTH'])[data.columns[0]].count().unstack()

FAUMON_ratio = fault_num4 / fault_num4.sum()
FAUMON_ratio_T = FAUMON_ratio.T

clusters = fclusterdata(np.array(FAUMON_ratio_T), 0.70)
clusters = clusters+20
print clusters

clusters_mapping = {label:idx for label,idx in zip(FAUMON_ratio.columns, clusters)}


data['FAULT_MONTH'] = data['FAULT_MONTH'].map(clusters_mapping)
'''
#get_dummies
#name_fault_month, enc_fault_month = onehot_pre('FAULT_MONTH')

data['INST_MONTH'] = pd.Categorical(data['INST_MONTH'], ordered=True)
name_inst_month = onehot_pre('INST_MONTH')


# ## 故障月份-故障类型统计
# - 故障月份-故障数量统计表显示了不同月份故障数量的分布，分布不是很均匀
# - 故障月份-故障类型图显示了每月的故障类型分布情况，每个月的故障类型占比基本相似，是比较弱的分类特征。
# - 故障月份-故障类型分布图，不同月份故障类型占比基本相似。

# In[22]:

data['COMM_MODE'].value_counts()
#plot_fun('COMM_MODE', 'FAULT_TYPE')
COMM_freq = data['COMM_MODE'].value_counts().index[data['COMM_MODE'].value_counts().values<100]
data['COMM_MODE'] = data['COMM_MODE'].replace(COMM_freq.values, 0)
name_comm_mode = onehot_pre('COMM_MODE')


# In[23]:

data.columns


# In[27]:

from sklearn.model_selection import train_test_split
import gc
gc.enable()
gc.collect()
data.drop(['month', 'FAULT_DATE1', 'INST_DATE1', 'FAULT_MONTH', 'FAULT_TYPE_1'], axis=1, inplace=True)
data_y = data['FAULT_TYPE']
data.fillna(0, inplace=True)
#data.astype({'SYNC_ORG_NO':int,'ORG_NO':int, 'SPEC_CODE':int, 'COMM_MODE':int,'MANUFACTURER':int,'INST_MONTH':int})
enc = preprocessing.OneHotEncoder()
enc.fit(data.drop('FAULT_TYPE',axis=1))
train, test= train_test_split(data, test_size=0.4, random_state=27, stratify=data_y)
train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)
gc.collect()
l = []
for n in range(test.shape[0]):
    if len(np.where(np.all(np.array(test)[n] == np.array(train), axis=1))[0]):
        l.append(n)
    if (n%10000)==0:
        print n
print test.shape
test.drop(l, inplace=True)
print test.shape
train_y = train['FAULT_TYPE']
test_y = test['FAULT_TYPE']
train.drop(['FAULT_TYPE'], axis=1, inplace=True)
test.drop(['FAULT_TYPE'], axis=1, inplace=True)
train_dummies = enc.transform(train)
test_dummies = enc.transform(test)
del data, train, test


# In[25]:



# In[23]:


# ## 机器学习算法故障预测

# In[24]:

from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import GridSearchCV
import pickle
from scipy.sparse import coo_matrix

#encode label
le = preprocessing.LabelEncoder()
le.fit(data_y)
train_y = le.transform(train_y)
test_y = le.transform(test_y)


# In[28]:

import xgboost as xgb
from xgboost.sklearn import XGBClassifier

TRAIN = True  #是否训练
CV = False
#split train set and test set
dtrain = xgb.DMatrix(train_dummies, train_y)
dtest = xgb.DMatrix(test_dummies)

clf = xgb.XGBClassifier(
    learning_rate = 0.2,
    n_estimators = 660,
    max_depth = 8,
    colsample_bytree = 0.8,
    subsample = 0.9,
    objective = 'multi:softmax',
    min_child_weight = 1,
    gamma = 2,
    seed = 27
    )

param = clf.get_xgb_params()
param['num_class'] = 11
if CV:
    cvresult = xgb.cv(param, dtrain, num_boost_round=2000, nfold=3, stratified=True,
                  metrics='merror', early_stopping_rounds=10,verbose_eval=True)
    clf.set_params(n_estimators=cvresult.shape[0])   #set n_estimators as cv rounds
if TRAIN:
    clf.fit(train_dummies, train_y, eval_metric='merror')
else:
    clf = pickle.load(open("zhejiang_4_all.pkl", "rb"))


ypred_xgb = clf.predict(test_dummies)
ypred_xgb = le.inverse_transform(ypred_xgb)
test_y_xgb = le.inverse_transform(test_y)
#print model report:
print(classification_report(test_y_xgb, ypred_xgb))
print(confusion_matrix(test_y_xgb, ypred_xgb))

xgb.plot_importance(clf.booster())
pickle.dump(clf, open("zhejiang_4_all_nore.pkl", "wb"))


# * 召回率(Recall)=  系统检索到的相关文件 / 系统所有相关的文件总数
# * 准确率(Precision) =  系统检索到的相关文件 / 系统所有检索到的文件总数
# * f1 = 2*Recall*Precision / (Recall+Precision)

# In[ ]:
'''
param_test1 = {'max_depth':range(5,12,2), 'min_child_weight':range(1,7,2)}
gsearch1 = GridSearchCV(estimator=clf, param_grid = param_test1, scoring='accuracy',n_jobs=-1,cv=2, verbose=True)
gsearch1.fit(train, train_y)
print gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_


# In[ ]:

data['FAULT_TYPE'].value_counts()


# # SGDClassifier
# 使用随机梯度下降线性分类器

# 对于线性不可分情况，使用rbf核将数据映射到高维空间中

# In[ ]:

from sklearn.kernel_approximation import RBFSampler, Nystroem

USE_RBF = False   #True：RBFSampler, False:Nystroem
if USE_RBF:
    rbf_feature = RBFSampler(gamma=1, random_state=1)
    train_SGD = rbf_feature.fit_transform(train)
    test_SGD = rbf_feature.transform(test)
else:
    Nys_feature = Nystroem(gamma=1, random_state=1)
    train_SGD = Nys_feature.fit_transform(train)
    test_SGD = Nys_feature.transform(test)



# In[ ]:

from sklearn.linear_model import SGDClassifier

USE_GridSearch = False
clf = SGDClassifier(loss='modified_huber', alpha=0.01, n_iter=100, class_weight="balanced", random_state=27)
if USE_GridSearch:
    param_test1 = {'loss':['hinge', 'log','modified_huber', 'squared_hinge', 'perceptron'], 'alpha':[0.1, 0.01, 0.01, 0.0001]}
    gsearch1 = GridSearchCV(estimator=clf, param_grid = param_test1, scoring='accuracy', n_jobs=-1,cv=2, verbose=True)
    gsearch1.fit(train_SGD, train_y)
    print gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
    clf = gsearch1
else:
    clf.fit(train_SGD, train_y)
ypred_sgd = clf.predict(test_SGD)
ypred_sgd = le.inverse_transform(ypred_sgd)
test_y_sgd = le.inverse_transform(test_y)
#print model report:
print(classification_report(test_y_sgd, ypred_sgd))
print(confusion_matrix(test_y_sgd, ypred_sgd))
pickle.dump(clf, open("zhejiang_4_SGD.pkl", "wb"))


# ## KNN

# In[ ]:

from sklearn import neighbors

USE_GridSearch = False
n_neighbors = 50

clf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
if USE_GridSearch:
    param_test1 = {'n_neighbors':range(20,60,10), 'weights':['uniform', 'distance']}
    gsearch1 = GridSearchCV(estimator=clf, param_grid = param_test1, scoring='accuracy', n_jobs=-1,cv=2, verbose=True)
    gsearch1.fit(train, train_y)
    print gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
    clf = gsearch1
else:
    clf.fit(train, train_y)
ypred_knn = clf.predict(test)
ypred_knn = le.inverse_transform(ypred_knn)
test_y_knn = le.inverse_transform(test_y)
#print model report:
print(classification_report(test_y_knn, ypred_knn))
print(confusion_matrix(test_y_knn, ypred_knn))
pickle.dump(clf, open("zhejiang_4_KNN.pkl", "wb"))


# * XGBoost算法使用决策树作为弱分类器，如果训练数据可分的情况下将会一直拟合数据知道训练准确率100%，可以证明数据不可分
# * SGDClassifier是针对数据量较大的线性分类器，当线性分类无效时，使用rbf将数据映射到高维空间中，再采用线性分类其进行分类，效果不佳
# * 使用KNN算法
# ## 可能存在的问题
# 1. 是否可以将使用时长、故障月份作为故障类型的预测属性
# * 在真实测试中，在安装电表时，电表还未投入使用无法得到使用时长、故障月份数据。
# 2. 测试集如何划分才能有效的评测分类器性能好坏
# * 划分测试集的目的是使训练与测试集中的数据不同，以测试训练所得的分类器在未知数据上的泛化能力，如果有很大部分是相同的话划分训练测试的初衷是什么（超过一半）。
# * 由于相同属性会有不同类别的频次，这个信息可以被树类等机器学习算法学到，所以相同数据的重复数据在没有更加有区分度的属性之前暂时保留。
# * 划分测试与训练的基本原则是保持其同分布，由于测试数据从所有数据中抽取30%，是为了代表这类问题的普遍性。使用树类的决策树算法，可以无限拟合训练数据，对于这类的数据很可能机器只是记忆了数据，一个模型想要训练准确率高很容易，但是往往一个越简单的算法模型我们认为他具有更好的泛化能力。训练的数据可能在真实的情况中再出现，想要看模型对于训练数据的性能好坏，观察training曲线就可以，没有必要再加入到测试集中，一个好的分类器应该是在这类问题上的任何情况都能表现的很好，而把训练数据加入到测试中，就像把一道测试题的结果已经提前‘泄露’给了训练模型。我们要实现的是对与故障预测这一类问题的分类器，我们要从有限的历史数据中学习出一个泛化能力较好的分类器，训练数据只是大量历史与未来数据中的一部分，这个分类器在这个数据上表现怎么样并不能代表这类问题就能解决好，往往是他有了能够对未知数据较好的处理能力，就像学习一类题目，能举一反三，才能说这类问题我们解决了！

# In[ ]:
'''


