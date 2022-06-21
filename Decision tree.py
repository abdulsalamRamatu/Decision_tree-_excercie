#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[4]:


import pandas as pd

df=pd.read_csv('C:\\Users\\Ramatu\\.jupyter\\py-master\\ML\\9_decision_tree\\Exercise\\titanic.csv')
df.head()


# In[5]:


data= df.drop(['PassengerId','Name', 'SibSp', 'Parch', 'Ticket', 'Cabin','Embarked'], axis='columns')


# In[6]:


data.head()


# In[7]:


target=df['Survived']


# In[8]:


from sklearn.preprocessing import LabelEncoder
lc_sex=LabelEncoder()


# In[9]:


data['Sex']=lc_sex.fit_transform(data['Sex'])


# In[14]:


data.head()


# In[10]:


from sklearn import tree
import math


# In[11]:


model= tree.DecisionTreeClassifier()
from sklearn.model_selection import train_test_split


# In[13]:





# In[31]:


data.Age=data.Age.fillna(data.Age.mean())
data.Age[:9]


# In[33]:


data_train,data_test,target_train, target_test=train_test_split(data,target,test_size=0.2)
len(data_train)


# In[34]:


import numpy as np
data.replace([np.inf, -np.inf], np.nan, inplace=True)
model.fit(data_train,target_train)


# In[35]:


model.score(data_test, target_test)


# In[36]:


len(data_test)


# In[ ]:




