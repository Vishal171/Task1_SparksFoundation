#!/usr/bin/env python
# coding: utf-8

# # The Sparks Foundation - Data Science & Business Analytics Internship
# ## Author:Vishal Gupta
# ## Task 1: Prediction Using Supervised Machine Learning
# ##### In this task it is required to predict the percentage of a student on the basis of number of hours studied using the Linear regression supervised Machine Learning algorithm

# ##### Step 1: Importing the essential packages used to solve the problem

# In[1]:


# To import and read the dataset
import pandas as pd
import numpy as np
# For plotting the dataset
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[3]:


# Readind Data from related link

url = "http://bit.ly/w-data"
df = pd.read_csv(url)


# In[5]:


# Observation of the Data
df.head()


# In[6]:


# To find number of rows and columns of the dataset
df.shape


# In[7]:


# To find information about the dataset
df.info()


# In[8]:


df.describe()


# In[9]:


# To check if the dataset contains null or missing values 
df.isnull().sum()


# ##### Step2:Visualizing the dataset

# In[54]:


# Pllotting the dataset
plt.figure(figsize=(16,9))
df.plot(x='Hours',y='Scores',style='.',color='blue',markersize=5)
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.grid()
plt.show()


# ##### From the above graph we can observe that there is a linear relation between 'Hours Studied' and 'Percentage score'.So we can use the linear regression model to predict future values

# In[15]:


# To determine the corelation among the data we can also use .corr method
df.corr()


# ##### Step3:Data Preparation
# ###### In this step the data is divided into 'Features' (inputs) and 'Labels' (outputs).After that the whole dataset is split into 2 parts testing data and training data.

# In[16]:


df.head()


# In[39]:


# Using the iloc function to divide the data
X = df.iloc[:,:1].values
Y = df.iloc[:,1:].values


# In[40]:


X


# In[41]:


Y


# In[42]:


# Splitting the data into testing data and training data 


from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)


# ##### Step4:Training the Algorithm
# ###### The data is splitted and now the model will be trained 

# In[43]:


from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train,Y_train)


# ##### Step5:Visualizing the Model

# In[44]:


line = model.coef_*X + model.intercept_
# Plotting for the training data
plt.figure(figsize=(16,9))
plt.scatter(X_train,Y_train,color='red')
plt.plot(X,line,color='green');
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.grid()
plt.show()


# In[45]:


# Plotting for the testing data
plt.figure(figsize=(16,9))
plt.scatter(X_test,Y_test,color='red')
plt.plot(X,line,color='green')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.grid()
plt.show()


# ##### Step6:Making Predictions

# In[46]:


print(X_test)#Testing data in Hours
Y_pred = model.predict(X_test) # Predicting the scores


# In[47]:


# Compairing Actual vs Predicted 

Y_test


# In[48]:


Y_pred


# In[49]:


# Compairing Actual vs Predicted 
comp = pd.DataFrame({'Actual':[Y_test],'Predicted':[Y_pred]})
comp


# In[52]:


# Testing with own data 
hours = 9.25
own_pred = model.predict([[hours]])
print('The predicted score if a person studies for',hours,'hours is',own_pred[0])


# ##### Step7:Evaluating the model
# 

# In[53]:


from sklearn import metrics
print('Mean Absolute Error:',metrics.mean_absolute_error(Y_test,Y_pred))


# In[ ]:




