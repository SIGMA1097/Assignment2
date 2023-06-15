#!/usr/bin/env python
# coding: utf-8

# # Assignment #2 (Git, GitHub, ML) 
# 
# 
# Name : Sigma Sasidharan
# Id : 100846417

# In[10]:


#Load Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[11]:


#Loading data from csv
x1=pd.read_csv('data.csv')
x1.head(10)


# In[12]:


x1 = x1.drop('Unnamed: 32', axis=1)


# In[13]:


x1.head(10)


# In[14]:


#show key stx1
x1.describe().T


# In[15]:


#Identify number of classes
x1.diagnosis.unique()


# In[16]:


#Create x and y variables
x = x1.drop('diagnosis',axis=1).to_numpy()
y = x1['diagnosis'].to_numpy()


# In[17]:


#Create Train and Test datasets
from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y,test_size = 0.20,random_state=100)

#Scale the data
from sklearn.preprocessing import StandardScaler  
sc = StandardScaler()  
x_train2 = sc.fit_transform(x_train)
x_test2 = sc.transform(x_test)


# In[18]:


#Script for Logistical Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix  

for name,method in [('Logistic Regression', LogisticRegression(solver='liblinear',random_state=100))]: 
    method.fit(x_train2,y_train)
    predict = method.predict(x_test2)
    #target_names=['M','B']
    print(confusion_matrix(y_test,predict))  
    #print(classification_report(y_test,predict,target_names=target_names))
    print(classification_report(y_test,predict,zero_division=0))


# In[ ]:





# In[ ]:




