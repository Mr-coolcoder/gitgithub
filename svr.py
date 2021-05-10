#!/usr/bin/env python
# coding: utf-8

# In[260]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[261]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR


# In[262]:


filepath = ('C:/Users/USER/Desktop/coursera/ML/tutorial/machine-learning-ex1/ex1/ex1data1.txt')
data =pd.read_csv(filepath,sep=',',header=None)
X = data.values[:,:1]
y = data.values[:,1:2]


# In[263]:


X = X.reshape(-1,1)
np.ravel(y)
scale = StandardScaler()
X = scale.fit_transform(X)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.2,random_state=0)


# In[264]:


regressor = SVR(kernel = 'poly')
regressor.fit(X_train, np.ravel(y_train,order='C'))


# In[265]:


#regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
#regr.fit(X,  np.ravel(y,order='C'))


# In[266]:


y_Pred = regressor.predict(X)


# In[267]:


plt.scatter(X,y,color="black")
plt.plot(X,y_Pred,color="yellow",label = "SVR Model with poly kernel")
plt.xlabel("Population of city in 10,000s")
plt.ylabel("profit in $10,000s")
plt.legend()


# In[268]:


regressor.score(X,y)


# In[ ]:




