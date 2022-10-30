import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

test=pd.read_csv("test_data.csv",error_bad_lines=False)
train=pd.read_csv("training_data.csv",error_bad_lines=False)

test.head()
train.head()
train.info()
train=train.drop('Unnamed: 133',axis=1)
train.head()

y_train=train.prognosis
x_train=train.drop('prognosis',axis=1)
x_train
y_train

y_test=test.prognosis
x_test=test.drop('prognosis',axis=1)
x_test.head()
y_test.head()

f,ax = plt.subplots(figsize=(75,16))
sns.countplot(y_train,label="Count",ax=ax) 




from xgboost import XGBClassifier
xgb = XGBClassifier(base_score=0.5,booster='gbtree', max_depth=10)
xgb.fit(x_train,y_train)
   

pickle.dump(xgb,open('disease.pkl','wb'))

col=x_train.columns
type(col)
len(col)

inputt = "itching stomach_pain skin_rash  fever cough".split(' ')
inputt

b=[0]*len(col)
for x in range(0,132):
    for y in inputt:
        if(col[x]==y):
            b[x]=1
b=np.array(b)
b=b.reshape(1,132)
sol=xgb.predict(b)
sol

