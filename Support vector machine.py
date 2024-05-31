#!/usr/bin/env python
# coding: utf-8

# In[29]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
digits=datasets.load_digits()


# In[30]:


iris=datasets.load_iris()


# In[31]:


df=pd.DataFrame(data=np.c_[iris['data'],iris['target']],columns=iris['feature_names']+['target'])


# In[32]:


df.head()


# In[33]:


df["target"].value_counts()


# In[34]:


#fit a SVM model to the data
from sklearn.svm import SVC
model=svm.SVC(kernel="linear",C=1)
model.fit(iris.data,iris.target)


# In[35]:


model.score(iris.data,iris.target)


# In[40]:


#make prediction
expected=iris.target
predicted=model.predict(iris.data)


# In[41]:


#summarise the fit of the model
print(metrics.classification_report(expected,predicted))
print(metrics.confusion_matrix(expected,predicted))


# In[71]:


#Tuning parameters
x=iris.data[:, :2]
y=iris.target


# In[97]:


def MySVMClassifier(my_kernel,my_C,my_gamma):
    svc=svm.SVC(kernel=my_kernel,C=my_C,gamma=my_gamma)
    svc.fit(x,y)
    h=0.02
    x_min,x_max=x[:, 0].min() -1,x[:, 0].max()+1
    y_min,y_max=x[:, 1].min() -1,x[:, 1].max()+1
    xx, yy =np.meshgrid(np.arange(x_min,x_max,h))
    
    plt.subplot(1,1,1)
    z=svc.predict(np.c_[xx.ravel(),yy.ravel()])
    z=z.reshape(xx.shape)
    plt.contourf(x[:, 0], x[:, 1], c=y)
    plt.xlabel("Sepal length")
    plt.ylabel("sepal width")
    plt.xlim(xx.min(),xx.max())
    plt.show()


# In[98]:


from ipywidgets import interact


# In[100]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from ipywidgets import interact

# Load data
data = load_iris()
X, y = data.data[:, :2], data.target  # Take only the first two features for visualization purposes

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def MySVMClassifier(my_kernel, my_C, my_gamma):
    svc = svm.SVC(kernel=my_kernel, C=my_C, gamma=my_gamma)
    svc.fit(X_train, y_train)
    
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    plt.figure(figsize=(10, 6))
    z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    plt.contourf(xx, yy, z, alpha=0.8)
    
    # Plot the training points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.xlabel("Sepal length")
    plt.ylabel("Sepal width")
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(f'SVM with {my_kernel} kernel, C={my_C}, Gamma={my_gamma}')
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.show()

# Create interactive widget
interact(MySVMClassifier, 
         my_kernel=["linear", "rbf"],
         my_C=(0.001, 10),
         my_gamma=(1, 100))


# In[ ]:




