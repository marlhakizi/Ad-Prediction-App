
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os
#get_ipython().run_line_magic('matplotlib', 'inline')
#f=open('/Users/marlynehakizimana/Downloads/train_NA17Sgz/train.csv')
#g=open('/Users/marlynehakizimana/Downloads/train_NA17Sgz/item_data.csv')
#s=open('/Users/marlynehakizimana/Downloads/train_NA17Sgz/view_log.csv')
#First 10 rows
train=pd.read_csv('train.csv',sep=',')
items=pd.read_csv('item_dat.csv',sep=',')
log=pd.read_csv('logs.csv',sep=',')



item_log=log.merge(items,how='left',on='item_id')
train3=train.merge(item_log,how='left',on='user_id')


colcol=['impression_id', 'impression_time','user_id', 'app_code', 'os_version','is_4G','is_click']
alltrain=train3[colcol].drop_duplicates()

cat_agg=['count','nunique']
num_agg=['min','mean','max','sum']
agg_col={'server_time':'nunique',
    'session_id':'nunique','item_price':'mean',
       'category_3':['nunique','mean'], 'product_type':['nunique','mean']
}

for k in train3.columns:
    if k.startswith('category_1') or k.startswith('category_2'):
        agg_col[k]=['sum','mean']
    elif k.startswith('cumcount'):
        agg_col[k]=num_agg


untrain=train3.groupby('impression_id').agg(agg_col)
untrain.columns=['J_' + '_'.join(col).strip() for col in untrain.columns.values]
on=untrain.reset_index()
allall=on.merge(alltrain,how='left',on='impression_id')


r=allall.groupby(['app_code']).count().reset_index()
aa=r[r.impression_id<1000].app_code
#new dataset with all app_codes less thn
allallu=allall[allall.app_code.isin(aa)]



allallu.loc[:,'impression_time']=pd.to_datetime(allallu['impression_time'])
allallu['Hour']=allallu.loc[:,'impression_time'].dt.hour
allallu['Day']=allallu.loc[:,'impression_time'].dt.day


allallu['newHour']=pd.cut(allallu.Hour,bins=[0,6,12,17,23],labels=['Early','Morning','Afternoon','Night'],include_lowest=True)


hou=pd.get_dummies(allallu.newHour)
ensemble=pd.concat([allallu,hou],axis=1)


rty=['impression_id', 'item_id','impression_time','user_id','Hour','os_version',
       'server_time_y', 'impression_id_y','newHour','is_click']

rtrt=[i for i in ensemble.columns if i not in rty]
ensemble1=ensemble[rtrt].fillna(0)


from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,classification_report,r2_score,roc_curve
from math import sqrt
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score, precision_score,f1_score,roc_auc_score, roc_curve
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import GridSearchCV

X=ensemble1
Y=ensemble['is_click']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,random_state=0)
os = SMOTE(random_state=0)
columns = X_train.columns
os_data_X,os_data_y=os.fit_sample(X_train, Y_train)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=['is_click'])

parameters={'n_estimators':[200,300],'max_depth':[5,10],}
rando=RandomForestClassifier(random_state=None)
clf = GridSearchCV(rando, parameters, cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=1994))
clf.fit(os_data_X, np.ravel(os_data_y,order='C'))

#clf.best_score_
from sklearn.externals import joblib
joblib.dump(clf, 'model.pkl')

# Load the model that you just saved
clf = joblib.load('model.pkl')

# Saving the data columns from training
model_columns = list(X.columns)
joblib.dump(model_columns, 'model_columns.pkl')
print("Models columns dumped!")
