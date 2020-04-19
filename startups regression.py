#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
df=pd.read_csv("/Users/adityasarkar/Desktop/50_Startups.csv")
#preprocessing
x=df[['R&D Spend','Administration','Marketing Spend','State']].values
y=df[['Profit']].values
ct=ColumnTransformer(transformers=[("oh", OneHotEncoder(), [3])],remainder='passthrough')
x=ct.fit_transform(x)
X=pd.DataFrame(x)
Y=pd.DataFrame(y)
#regression model
reg=linear_model.LinearRegression()
reg.fit(X,Y)
#enter values
reg.predict([])


# In[3]:


cd Desktop


# In[ ]:




