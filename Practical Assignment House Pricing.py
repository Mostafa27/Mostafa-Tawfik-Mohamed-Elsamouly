#!/usr/bin/env python
# coding: utf-8

# In[534]:


# used for manipulating directory paths
import os

# Scientific and vector computation for python
import numpy as np

# Plotting library
from matplotlib import pyplot

import pandas as pd


# In[577]:


data= pd.read_csv("house_data_complete.csv")


# In[831]:


train, validate, test = np.split(data.sample(frac=1), [int(.6*len(data)),int(.8*len(data))])


# In[832]:


train


# In[833]:


validate


# In[834]:


test


# In[835]:


pyplot.plot(train.values[:,3], train.values[:,2], 'ro', ms=10, mec='k')
pyplot.ylabel('Price')
pyplot.xlabel('Bedrooms')


# In[836]:


m= train.values[:,2].size
def featureNormalize(X):
    X_norm = X.copy()
    mu = np.zeros(X.shape[1])
    sigma = np.zeros(X.shape[1])
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)    
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma


# In[837]:


X = train.drop(columns=['price', 'date']).values
X_norm, mu, sigma = featureNormalize(X)
print('Computed mean:', mu)
print('Computed standard deviation:', sigma)


# In[838]:


X = np.concatenate([np.ones((m, 1)), X_norm], axis=1)


# In[839]:


initial_theta = np.zeros(X.shape[1])
h1 = np.dot(X, initial_theta)
y = train.values[:,2]
lambda_ = 1


# In[840]:


def costFunctionReg(theta, X, y, h, lambda_):
    m= train.values[:,2].size
    J= np.dot((h - y), (h - y)) / (2 * m) + ((lambda_/(2 * m))* np.sum(np.dot(theta, theta)))
    return J


# In[841]:


cost = costFunctionReg(initial_theta, X, y, h1, lambda_)

print('Cost at initial theta (zeros): {:.3f}'.format(cost))


# In[842]:


def gradientDescent(X, y, theta, alpha, num_iters):
    m = y.shape[0]  
    theta = theta.copy()
    J_history = []  

    for i in range(num_iters):
        h = np.dot(X, theta)
        theta = theta*(1 - (alpha*lambda_)/m) - ((alpha / m) * (np.dot(X.T, h - y)))
        J_history.append(costFunctionReg(theta, X, y, h, lambda_))

    return theta, J_history


# In[843]:


iterations = 100
alpha = 0.01
theta, J_history = gradientDescent(X,y, initial_theta, alpha, iterations)
print(theta)
print(J_history)


# In[844]:


pyplot.plot(np.arange(len(J_history)), J_history, lw=2)
pyplot.xlabel('Number of iterations')
pyplot.ylabel('Cost J')


# In[845]:


def gradientDescent2(X, y, theta, alpha, num_iters):
    m = y.shape[0]  
    theta = theta.copy()
    J_history = []  

    for i in range(num_iters):
        h = np.dot(np.power(X,2), theta)
        theta = theta*(1 - (alpha*lambda_)/m) - ((alpha / m) * (np.dot(X.T, h - y)))
        J_history.append(costFunctionReg(theta, X, y, h, lambda_))

    return theta, J_history


# In[846]:


iterations = 100
alpha = 0.003
theta2, J_history2 = gradientDescent2(X,y, initial_theta, alpha, iterations)
print(theta2)
print(J_history2)


# In[847]:


pyplot.plot(np.arange(len(J_history2)), J_history2, lw=2)
pyplot.xlabel('Number of iterations')
pyplot.ylabel('Cost J')


# In[848]:


def gradientDescent3(X, y, theta, alpha, num_iters):
    m = y.shape[0]  
    theta = theta.copy()
    J_history = []  
    k = X.copy()
    k[:, 4] = np.power(k[:, 4], 2)
    for i in range(num_iters):
        h = np.dot(k, theta)
        theta = theta*(1 - (alpha*lambda_)/m) - ((alpha / m) * (np.dot(X.T, h - y)))
        J_history.append(costFunctionReg(theta, X, y, h, lambda_))

    return theta, J_history


# In[849]:


iterations = 100
alpha = 0.01
theta3, J_history3 = gradientDescent3(X,y, initial_theta, alpha, iterations)
print(theta3)
print(J_history3)


# In[850]:


pyplot.plot(np.arange(len(J_history3)), J_history3, lw=2)
pyplot.xlabel('Number of iterations')
pyplot.ylabel('Cost J')


# In[851]:


pyplot.plot(np.arange(len(J_history)), J_history, lw=2, label='h1')
pyplot.plot(np.arange(len(J_history2)), J_history2, lw=2, label='h2')
pyplot.plot(np.arange(len(J_history3)), J_history3, lw=2, label='h3')
pyplot.xlabel('Number of iterations')
pyplot.ylabel('Cost J')
pyplot.legend()


# In[ ]:




