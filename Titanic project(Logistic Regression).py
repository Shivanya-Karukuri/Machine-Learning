#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
titanic_data=pd.read_csv(r"D:\EXCEL\titanic.csv")
titanic_data.head(10)

titanic_data.head(10)


# In[2]:


print("the number of passangers",str(len(titanic_data)))


# In[3]:


import seaborn as sns
sns.countplot(x="Survived",data=titanic_data)


# In[4]:


sns.countplot(x="Survived",hue="Sex",data=titanic_data)


# In[5]:


sns.countplot(x="Survived",hue="Pclass",data=titanic_data)


# In[6]:


titanic_data["Age"].plot.hist()


# In[7]:


titanic_data["Fare"].plot.hist()


# In[8]:


sns.countplot(x="SibSp",data=titanic_data)


# In[9]:


titanic_data.isnull()


# In[10]:


titanic_data.isnull().sum()


# In[11]:


sns.countplot(x="SibSp",data=titanic_data)


# In[12]:


sns.heatmap(titanic_data.isnull(),yticklabels=False,cmap="viridis")


# In[13]:


sns.boxplot(x="Pclass",y="Age",data=titanic_data)


# In[14]:


titanic_data.drop("Cabin",axis=1,inplace=True)


# In[15]:


titanic_data.head(5)


# In[16]:


titanic_data.dropna(inplace=True)


# In[17]:


titanic_data.head(5)


# In[18]:


titanic_data.isnull().sum()


# In[19]:


sns.heatmap(titanic_data.isnull())


# In[20]:


sex=pd.get_dummies(titanic_data["Sex"],drop_first=True)
sex


# In[21]:


embark=pd.get_dummies(titanic_data["Embarked"],drop_first=True)
embark


# In[22]:


pcl=pd.get_dummies(titanic_data["Pclass"],drop_first=True)
pcl


# In[23]:


titanic_data=pd.concat([titanic_data,sex,embark,pcl],axis=1)
titanic_data.head(5)


# In[24]:


print(titanic_data.columns)


# In[25]:


titanic_data.head()


# In[26]:


titanic_data.drop(["Sex","Embarked","PassengerId","Name","Ticket"],axis=1,inplace=True)


# In[30]:


titanic_data.head()


# In[33]:


#Train the data
x=titanic_data.drop('Survived',axis=1)
y=titanic_data["Survived"]


# In[34]:


from sklearn.model_selection import train_test_split


# In[35]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)


# In[36]:


from sklearn.linear_model import LogisticRegression


# In[37]:


logmodel = LogisticRegression()
logmodel.fit(x_train, y_train)


# In[38]:


predictions=logmodel.predict(x_test)


# In[39]:


from sklearn.metrics import classification_report


# In[40]:


classification_report(y_test,predictions)


# In[41]:


from sklearn.metrics import confusion_matrix


# In[42]:


confusion_matrix(y_test,predictions)


# In[43]:


from sklearn.metrics import accuracy_score


# In[44]:


accuracy_score(y_test,predictions)


# In[ ]:




