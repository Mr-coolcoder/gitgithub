#!/usr/bin/env python
# coding: utf-8

# In[147]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[148]:


filepath = ('C:/Users/USER/Desktop/coursera/ML/tutorial/machine-learning-ex1/ex1/ex1data1.txt')
data =pd.read_csv(filepath,sep=',',header=None)
X = data.values[:,:1]
y = data.values[:,1:2]


# In[149]:


X = X.reshape(-1,1)
scale = StandardScaler()
X = scale.fit_transform(X)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.2,random_state=0)


# In[150]:


poly = PolynomialFeatures(degree=2)


# In[151]:


X_poly = poly.fit_transform(X_train)


# In[152]:


poly.fit(X_poly,y_train)


# In[153]:


linreg = LinearRegression()


# In[154]:


linreg.fit(X_poly,y_train)


# In[155]:


y_Pred = linreg.predict(poly.fit_transform(X))


# In[156]:


plt.scatter(X,y,color="black")
plt.plot(X,y_Pred,color="yellow",label = "polynomial regression Model with degree = 2")
plt.xlabel("Population of city in 10,000s")
plt.ylabel("profit in $10,000s")
plt.legend()


# In[157]:


linreg.score(poly.fit_transform(X),y)


# In[ ]:





# In[ ]:




