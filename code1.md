

```python
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier
np.random.seed(0)
```


```python
#Loading data
df_train = pd.read_csv('./input/train_users_2.csv')
df_test = pd.read_csv('./input/test_users.csv')
#extract destination of training set
#labels=df_train['country_destination'].values
```


```python
#df_train = df_train.drop(['country_destination'],axis=1)
#id_test = df_test['id'] #id of test set
df_train['tr'] = 'train'
df_test['tr'] = ' test'
```


```python
#Creating a DataFrame with train+test data
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True) #275,547
df_all = df_all.drop(['date_first_booking'],axis=1)
```


```python
#check missing values, data_first_booking,age have the most missing values
users_nan = (df_all.isnull().sum()*1.0 / df_train.shape[0]) * 100
users_nan
```




    affiliate_channel           0.000000
    affiliate_provider          0.000000
    age                        54.750739
    country_destination        29.091454
    date_account_created        0.000000
    first_affiliate_tracked     2.850771
    first_browser               0.000000
    first_device_type           0.000000
    gender                      0.000000
    id                          0.000000
    language                    0.000000
    signup_app                  0.000000
    signup_flow                 0.000000
    signup_method               0.000000
    timestamp_first_active      0.000000
    tr                          0.000000
    dtype: float64




```python
df_all['age']=df_all['age'].fillna(-1)
df_all['first_affiliate_tracked']=df_all['first_affiliate_tracked'].fillna(-1)
df_all.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 275547 entries, 0 to 275546
    Data columns (total 16 columns):
    affiliate_channel          275547 non-null object
    affiliate_provider         275547 non-null object
    age                        275547 non-null float64
    country_destination        213451 non-null object
    date_account_created       275547 non-null object
    first_affiliate_tracked    275547 non-null object
    first_browser              275547 non-null object
    first_device_type          275547 non-null object
    gender                     275547 non-null object
    id                         275547 non-null object
    language                   275547 non-null object
    signup_app                 275547 non-null object
    signup_flow                275547 non-null int64
    signup_method              275547 non-null object
    timestamp_first_active     275547 non-null int64
    tr                         275547 non-null object
    dtypes: float64(1), int64(2), object(13)
    memory usage: 35.7+ MB


#####Feature engineering#######


```python
##filling missing values
#set missing/weird values in age to -1
df_all.loc[df_all.age>95,'age'] = -1
df_all.loc[df_all.age<13,'age'] = -1
df_all['age'] = df_all['age'].astype(int)
```


```python
#make bins for age, the min is 15, max is 95, set bin size as 5,missing as one bin, 17 bins in total
bins = np.linspace(15,95,16)
df_all['agebin']=np.digitize(df_all.age,bins)
df_all['agebin'][df_all.age==-1] = -1
df_all['agebin'] = df_all['agebin']+1
```

    /Users/yanx/anaconda/lib/python2.7/site-packages/ipykernel/__main__.py:4: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy



```python
#date_account_created
dac = np.vstack(df_all.date_account_created.astype(str).apply(lambda x: list(map(int, x.split('-')))).values)
df_all['dac_year'] = dac[:,0]
df_all['dac_month'] = dac[:,1]
df_all['dac_day'] = dac[:,2]
df_all = df_all.drop(['date_account_created'], axis=1)

#timestamp_first_active
tfa = np.vstack(df_all.timestamp_first_active.astype(str).apply(lambda x: list(map(int, [x[:4],x[4:6],x[6:8],x[8:10],x[10:12],x[12:14]]))).values)
df_all['tfa_year'] = tfa[:,0]
df_all['tfa_month'] = tfa[:,1]
df_all['tfa_day'] = tfa[:,2]
df_all = df_all.drop(['timestamp_first_active'], axis=1)
```


```python
#One-hot-encoding features
ohe_feats = ['agebin']
for f in ohe_feats:
    df_all_dummy = pd.get_dummies(df_all[f], prefix=f)
    df_all = df_all.drop([f], axis=1)
    df_all = pd.concat((df_all, df_all_dummy), axis=1)
```


```python
#One-hot-encoding features
ohe_feats = ['dac_year','dac_month','dac_day','tfa_year','tfa_month','tfa_day','gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']
for f in ohe_feats:
    df_all_dummy = pd.get_dummies(df_all[f], prefix=f)
    df_all = df_all.drop([f], axis=1)
    df_all = pd.concat((df_all, df_all_dummy), axis=1)
```


```python
# keep age or not?
```


```python
#session data
sessions = pd.read_csv('./input/sessions.csv')
#sessions.head()
sessions.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 10567737 entries, 0 to 10567736
    Data columns (total 6 columns):
    user_id          object
    action           object
    action_type      object
    action_detail    object
    device_type      object
    secs_elapsed     float64
    dtypes: float64(1), object(5)
    memory usage: 564.4+ MB



```python
sessions.apply(lambda x: x.nunique(),axis=0)
```




    user_id          135483
    action              359
    action_type          10
    action_detail       155
    device_type          14
    secs_elapsed     337661
    dtype: int64




```python
sessions.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>action</th>
      <th>action_type</th>
      <th>action_detail</th>
      <th>device_type</th>
      <th>secs_elapsed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>d1mm9tcy42</td>
      <td>lookup</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Windows Desktop</td>
      <td>319</td>
    </tr>
    <tr>
      <th>1</th>
      <td>d1mm9tcy42</td>
      <td>search_results</td>
      <td>click</td>
      <td>view_search_results</td>
      <td>Windows Desktop</td>
      <td>67753</td>
    </tr>
    <tr>
      <th>2</th>
      <td>d1mm9tcy42</td>
      <td>lookup</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Windows Desktop</td>
      <td>301</td>
    </tr>
    <tr>
      <th>3</th>
      <td>d1mm9tcy42</td>
      <td>search_results</td>
      <td>click</td>
      <td>view_search_results</td>
      <td>Windows Desktop</td>
      <td>22141</td>
    </tr>
    <tr>
      <th>4</th>
      <td>d1mm9tcy42</td>
      <td>lookup</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Windows Desktop</td>
      <td>435</td>
    </tr>
  </tbody>
</table>
</div>




```python
secs = sessions.groupby(['user_id'])['secs_elapsed'].sum().reset_index()
secs.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>secs_elapsed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>00023iyk9l</td>
      <td>867896</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0010k6l0om</td>
      <td>586543</td>
    </tr>
    <tr>
      <th>2</th>
      <td>001wyh0pz8</td>
      <td>282965</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0028jgx1x1</td>
      <td>297010</td>
    </tr>
    <tr>
      <th>4</th>
      <td>002qnbzfs5</td>
      <td>6487080</td>
    </tr>
  </tbody>
</table>
</div>




```python
#sessions.groupby(['action','action_type']).sum()
```


```python
#combine action within action_type
action_type = pd.pivot_table(sessions, index= ['user_id'],columns=['action_type'],values='action',aggfunc=len,
                             fill_value=0).reset_index()
action_type = action_type.drop(['booking_response'],axis=1)
action_type = action_type.rename(columns={'-unknown-':'action_unknown'})
#action_type.head()
action_type.shape
```




    (135478, 10)




```python
device_type = pd.pivot_table(sessions, index = ['user_id'],columns = ['device_type'],values = 'action',aggfunc=len,fill_value=0).reset_index()
device_type = device_type.drop(['Blackberry','Opera Phone','iPodtouch','Windows Phone'],axis=1)
device_type = device_type.rename(columns={'-unknown-':'device_unknown'})
device_type.shape
```




    (135483, 11)




```python
sessions_data = pd.merge(action_type,device_type,on='user_id',how='inner')
sessions_data = pd.merge(sessions_data,secs,on='user_id',how='inner')
#sessions_data.info()
```


```python
#One-hot-encoding features
ohe_feats=list(sessions_data.columns.values)
ohe_feats=ohe_feats[1:len(ohe_feats)-1]
for f in ohe_feats:
    sessions_data_dummy = pd.get_dummies(sessions_data[f], prefix=f)
    sessions_data = sessions_data.drop([f], axis=1)
    sessions_data = pd.concat((sessions_data, sessions_data_dummy), axis=1)
```


```python
sessions_data.shape
```




    (135478, 6556)




```python
##split data by sessions info

df_all1= pd.merge(df_all, sessions_data, left_on='id', right_on='user_id' ,how='inner' )
id1 = df_all1['id']
df_all2 = df_all[~df_all['id'].isin(id1)]
df_all1.shape, df_all2.shape
```




    ((135478, 6828), (140069, 272))




```python
df_all1.shape, df_all2.shape, df_all.shape
```




    ((135478, 6826), (140069, 271), (275547, 272))




```python
##split data into train, test for df_all1
dff_train1 = df_all1[df_all1['tr']=='train']
labels1 = dff_train1['country_destination'].values

dff_test1  = df_all1[df_all1['tr']=='test']
id_test1 = dff_test1['id'] #id of test set
dff_train1.drop(['id','user_id','tr','country_destination'],axis=1,inplace=True)
dff_test1.drop(['id','tr','country_destination'],axis=1,inplace=True)
```

    /Users/yanx/anaconda/lib/python2.7/site-packages/ipykernel/__main__.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    /Users/yanx/anaconda/lib/python2.7/site-packages/ipykernel/__main__.py:7: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy



```python
##split data into train, test for df_all2
dff_train2 = df_all2[df_all2['tr']=='train']
labels2 = dff_train2['country_destination'].values

dff_test2  = df_all2[df_all2['tr']=='test']
id_test2 = dff_test2['id']
dff_train2.drop(['id','user_id','tr','country_destination'],axis=1,inplace=True)
dff_test2.drop(['id','tr','country_destination'],axis=1,inplace=True)
```

    /Users/yanx/anaconda/lib/python2.7/site-packages/ipykernel/__main__.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    /Users/yanx/anaconda/lib/python2.7/site-packages/ipykernel/__main__.py:7: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy



```python
X1 = dff_train1.values
le = LabelEncoder()
y1 = le.fit_transform(labels1)   
X_test1 = dff_test1.values
```


```python
X2 = dff_train2.values
le = LabelEncoder()
y2 = le.fit_transform(labels2)   
X_test2 = dff_test2.values
```


```python
##cross validation
```


```python
#Classifier1
xgb = XGBClassifier(max_depth=6, learning_rate=0.3, n_estimators=25,
                    objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=0)                  
xgb.fit(X1, y1)
y_pred1 = xgb.predict_proba(X_test1)  


#Classifier2
xgb.fit(X2, y2)
y_pred2 = xgb.predict_proba(X_test2)  

y_pred=pd.concat(y_pred1,y_pred2,axis=0,ignore_index=True)
```


```python
#Taking the 5 classes with highest probabilities

ids = []  #list of ids
cts = []  #list of countries
for i in range(len(id_test)):
    idx = id_test[i]
    ids += [idx] * 5
    cts += le.inverse_transform(np.argsort(y_pred[i])[::-1])[:5].tolist()

```


```python

```


```python
#Generate submission
sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
sub.to_csv('output/sub.csv',index=False)

```


```python
#try: 1.linear/logistic : dimension reduction(pca) 2.svm 3.random forest
```


```python
#agb = pd.read_csv('./input/age_gender_bkts.csv')
```


```python

```
