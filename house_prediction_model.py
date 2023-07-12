#!/usr/bin/env python
# coding: utf-8

# In[64]:


import pandas as pd


# In[65]:


import numpy as np


# In[66]:


import seaborn as sns


# In[67]:


import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[68]:


HouseDF =pd.read_csv("house.csv")


# In[69]:


HouseDF.head()


# In[70]:


HouseDF.info()


# In[71]:


HouseDF


# In[72]:


HouseDF.describe()


# In[73]:


HouseDF.columns


# In[74]:


sns.pairplot(HouseDF)


# In[75]:


sns.heatmap(HouseDF.corr(),annot=True)


# In[76]:


X=HouseDF[['price', 'area', 'latitude', 'longitude', 'Bedrooms', 'Bathrooms',
       'Price_sqft']]

y=HouseDF['price']


# In[56]:


from sklearn.model_selection import train_test_split


# In[77]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.40, random_state=101)


# In[78]:


from sklearn.linear_model import LinearRegression


# In[79]:


lm = LinearRegression()


# In[80]:


lm.fit (X_train, y_train)


# In[81]:


coeff_df = pd.DataFrame (lm.coef_, X.columns, columns=['Coefficient'])


# In[82]:


coeff_df


# In[84]:


predictions = lm.predict (X_test)


# In[85]:


plt.scatter (y_test, predictions)


# In[86]:


sns.distplot((y_test-predictions),bins=50);


# In[ ]:




