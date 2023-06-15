#!/usr/bin/env python
# coding: utf-8

# # Assignment #2 (Git, GitHub, ML) 
# 
# 
# Name : Sigma Sasidharan
# Id : 100846417

# In[1]:


#Load Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#Loading data from csv
x1=pd.read_csv('data.csv')
x1.head(10)


# In[3]:


x1 = x1.drop('Unnamed: 32', axis=1)


# In[4]:


x1.head(10)


# In[5]:


#show key statistics
x1.describe().T


# In[6]:


#Identify number of classes
x1.diagnosis.unique()


# In[7]:


#Create x and y variables
x = x1.drop('diagnosis',axis=1).to_numpy()
y = x1['diagnosis'].to_numpy()


# In[8]:


#Create Train and Test datasets
from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y,test_size = 0.20,random_state=100)

#Scale the data
from sklearn.preprocessing import StandardScaler  
sc = StandardScaler()  
x_train2 = sc.fit_transform(x_train)
x_test2 = sc.transform(x_test)


# In[9]:


#Script to get SVM and NB
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix  

for name,method in [('SVM', SVC(kernel='linear',random_state=100)),
                    ('Naive Bayes',GaussianNB())]: 
    method.fit(x_train2,y_train)
    predict = method.predict(x_test2)
    #target_names=['M','B']
    print('\nEstimator: {}'.format(name)) 
    print(confusion_matrix(y_test,predict))  
    print(classification_report(y_test,predict,zero_division=0)) 


# In[ ]:




