#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os


# In[ ]:





# In[3]:


os.chdir('F:/Data Science/Santander ML')


# In[4]:


os.getcwd()


# In[5]:


df = pd.read_csv('F:/Data Science/Data Science Project/train.csv')
pd.options.display.max_columns = None


# In[6]:


df.head()


# In[7]:


df.shape


# In[8]:


def missing_val(df):
    miss = df.isnull().sum()
    percent = (df.isnull().sum()/df.isnull().count()*100)
    data_frame = pd.concat([miss,percent],axis = 1,keys = ['Missing_val','Percentage'])
    
    data_type=[]
    for col in df.columns:
        dtype = str(df[col].dtype)
        data_type.append(dtype)
    data_frame['Data_Type'] =data_type
    return (np.transpose(data_frame))
    


# In[9]:


missing_val(df)


# In[10]:


df.describe()


# In[11]:


plt.style.use('fivethirtyeight')
sns.countplot(df['target'])


# In[12]:


print('There are {}% target values with 1'.format(100*df['target'].value_counts()[1]/df.shape[0]))


# In[13]:


df_corr = df.drop(['target'],axis=1)


# In[14]:


df_corr = df_corr.corr()


# In[15]:


plt.figure(figsize=(20,20))
sns.heatmap(df_corr)


# In[57]:


X = df.drop(columns = ['target','ID_code'],axis = 1)
y = df['target']


# In[58]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


# In[18]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()


# In[19]:


from sklearn.model_selection import StratifiedKFold


# In[20]:


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

   
for train_index, test_index in skf.split(X, y):
    X_train_fold, X_test_fold = X_scaled[train_index], X_scaled[test_index]
    y_train_fold, y_test_fold = y[train_index], y[test_index]
    
print('Shape of X_train_fold :',X_train_fold.shape)
print('Shape of X_test_fold :',X_test_fold.shape)
print('Shape of y_train_fold :',y_train_fold.shape)
print('Shape of y_test_fold :',y_test_fold.shape) 


# In[21]:


lr.fit(X_train_fold,y_train_fold)


# In[22]:


# Prediction
y_prob = lr.predict_proba(X_test_fold)[:,1]

y_pred = np.where(y_prob > 0.5,1,0)#if geater than 0.5 replace with 1 else 0


lr.score(X_test_fold,y_pred)


# In[23]:


from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test_fold,y_pred))


# In[24]:


report = classification_report(y_test_fold,y_pred)
print(report)


# In[25]:


from sklearn.metrics import roc_auc_score
auc_roc = roc_auc_score(y_test_fold,y_pred)
auc_roc


# In[26]:


from sklearn.metrics import roc_curve, auc
fpr ,tpr ,threshold = roc_curve(y_test_fold,y_prob)
roc_auc = auc(fpr, tpr)
roc_auc


# In[27]:


plt.plot(fpr ,tpr, label ='AUC=%0.2f'% roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0,1], [0,1],linestyle ='--')


# ### Random Forest 

# In[34]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 100)
rf.fit(X_train_fold, y_train_fold)


# In[35]:


y_pred = rf.predict(X_test_fold)

y_prob = rf.predict_proba(X_test_fold)[:,1]


# In[36]:


print(confusion_matrix(y_test_fold,y_pred))


# In[37]:


print(classification_report(y_test_fold,y_pred))


# In[38]:


fpr ,tpr ,threshold = roc_curve(y_test_fold,y_prob)
roc_auc = auc(fpr, tpr)
roc_auc


# In[39]:


plt.plot(fpr ,tpr, label ='AUC=%0.2f'% roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0,1], [0,1],linestyle ='--')


# ### Decision Tree (Gini)

# In[40]:


from sklearn.tree import DecisionTreeClassifier
dtg = DecisionTreeClassifier(criterion = 'gini',random_state=100,max_depth=8,min_samples_leaf = 7)


# In[41]:


dtg.fit(X_train_fold,y_train_fold)


# In[42]:


y_pred = dtg.predict(X_test_fold)


# In[43]:


y_prob = dtg.predict_proba(X_test_fold)[:,1]


# In[44]:


print(confusion_matrix(y_test_fold,y_pred))
print(classification_report(y_test_fold,y_pred))


# In[45]:


fpr ,tpr ,threshold = roc_curve(y_test_fold,y_prob)
roc_auc = auc(fpr, tpr)
roc_auc


# In[46]:


plt.plot(fpr ,tpr, label ='AUC=%0.2f'% roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0,1], [0,1],linestyle ='--')


# ### Dicision Tree (Entropy)

# In[163]:


dte = DecisionTreeClassifier(criterion = 'entropy',random_state=100,max_depth =15,min_samples_leaf = 5)
dte.fit(X_train_fold,y_train_fold)
                             


# In[164]:


y_pred = dte.predict(X_test_fold)
y_prob = dte.predict_proba(X_test_fold)


# In[165]:


print(confusion_matrix(y_test_fold,y_pred))
print(classification_report(y_test_fold,y_pred))


# In[166]:


param_dict ={
    'criterion':['gini','entropy'],
    'max_depth':range(1,10),
    'min_samples_split':range(1,10),
    'min_samples_leaf':range(1,5)
}
    


# In[171]:


from sklearn.model_selection import GridSearchCV
dt = DecisionTreeClassifier()


# In[ ]:


grid = GridSearchCV(dt,
                   param_grid=param_dict,
                   cv=2,verbose=1,
                   n_jobs=-1)
grid.fit(X_train_fold,y_train_fold)


# In[ ]:


grid.best_params_


# In[ ]:


grid.best_estimator_


# In[ ]:


grid.best_score_


# ### Smote

# In[ ]:





# In[59]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.33,random_state=44)


# In[60]:


X_train.shape,y_train.shape,X_test.shape,y_test.shape


# In[61]:


from imblearn.over_sampling import SMOTE, RandomOverSampler

sm = SMOTE(random_state=44)

X_smote,y_smote = sm.fit_resample(X_train,y_train)


# In[62]:


lr = LogisticRegression(random_state=44,solver='lbfgs',max_iter=1000)


# In[63]:


lr.fit(X_smote,y_smote)


# In[65]:


y_pred = lr.predict(X_test)

y_prob = lr.predict_proba(X_test)[:,1]


# In[66]:


print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[ ]:


fpr ,tpr ,threshold = roc_curve(y_test_fold,y_prob)
roc_auc = auc(fpr, tpr)
roc_auc
plt.plot(fpr ,tpr, label ='AUC=%0.2f'% roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0,1], [0,1],linestyle ='--')


# In[68]:


fpr, tpr, threshold = roc_curve(y_test,y_prob)
roc_auc = auc(fpr,tpr)
roc_auc

plt.plot(fpr,tpr, label = 'AUC=%0.2f'% roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0,1],[0,1],linestyle = '--')


# ### Decision Tree

# In[77]:



dfg = DecisionTreeClassifier(criterion ='entropy',random_state=100,max_depth=8,min_samples_leaf =7)


# In[78]:


dfg.fit(X_smote,y_smote)


# In[79]:


y_pred = dfg.predict(X_test)

y_prob = dfg.predict_proba(X_test)[:,1]


# In[80]:


print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[81]:


fpr,tpr,threshold = roc_curve(y_test,y_prob)
roc_auc = auc(fpr,tpr)
roc_auc

plt.plot(fpr,tpr, label='AUC%0.2f'% roc_auc)
plt.plot([0,1],[0,1],linestyle='--')
plt.legend(loc ='lower right')


# ### Light GBM

# In[83]:


import lightgbm as lgb


# In[86]:


lgb_train = lgb.Dataset(X_train_fold,label=y_train_fold)

lgb_test = lgb.Dataset(X_test_fold, label=y_test_fold)


# In[87]:


#Selecting best hyperparameters by tuning of different parameters:-
params={'boosting_type': 'gbdt', 
          'max_depth' : -1, #no limit for max_depth if <0
          'objective': 'binary',
          'boost_from_average':False, 
          'nthread': 20,
          'metric':'auc',
          'num_leaves': 50,
          'learning_rate': 0.01,
          'max_bin': 100,      #default 255
          'subsample_for_bin': 100,
          'subsample': 1,
          'subsample_freq': 1,
          'colsample_bytree': 0.8,
          'bagging_fraction':0.5,
          'bagging_freq':5,
          'feature_fraction':0.08,
          'min_split_gain': 0.45, #>0
          'min_child_weight': 1,
          'min_child_samples': 5,
          'is_unbalance':True,
          }


# In[88]:


num_rounds = 10000
lgbm = lgb.train(params,lgb_train,num_rounds,
                 valid_sets=[lgb_train,lgb_test],
                 verbose_eval=1000,early_stopping_rounds=5000)


# In[115]:


l = lgbm.predict(X_test_fold)


# In[116]:


l2= np.where(l>=0.5,1,0)
l2


# In[118]:


print(confusion_matrix(y_test_fold,l2))
print(classification_report(y_test_fold,l2))


# In[ ]:





# In[93]:


df_test = pd.read_csv('F:/Data Science/Data Science Project/test.csv')
df_test.head()


# In[ ]:





# In[94]:


X_test = df_test.drop(['ID_code'],axis=1)

lgbm_predict_prob = lgbm.predict(X_test,random_state =44,
                                 num_iteration=lgbm.best_iteration)

lgbm_predict=np.where(lgbm_predict_prob>=0.5,1,0)
print(lgbm_predict_prob)
print(lgbm_predict)


# In[91]:


lgb.plot_importance(lgbm,max_num_features=50,
                     importance_type='split',figsize=(20,50))


# In[125]:


df_sub = pd.DataFrame({'ID_code':df_test['ID_code'].values})
df_sub['lgbm_predict_prob'] = lgbm_predict_prob
df_sub['lgbm_predict'] = lgbm_predict
df_sub.to_csv('sub.csv',index=True)
df_sub.head()


# In[ ]:




