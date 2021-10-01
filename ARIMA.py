#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system(' pip install pmdarima')


# In[3]:


from pmdarima.arima import auto_arima

import torch


# In[4]:


X = torch.load('/home/vahidn/projects/def-banire/vahidn/training/X1.pt', map_location=torch.device('cpu'))
Y = torch.load('/home/vahidn/projects/def-banire/vahidn/training/y1.pt', map_location=torch.device('cpu'))


# In[5]:


import numpy as np

train = X.detach().numpy()
valid1 = Y.detach().numpy()
valid = valid1[:,:-4]


# In[ ]:


MSES = 0

for i in range(len(train)):

  print(i+1)
  train_u = train[i]
  valid_u = valid[i]
  model = auto_arima(train_u, start_p=1, max_p= 10, max_q=10, start_P=0, 
                    start_Q=0, max_P=10, max_Q=10, m=12, trace=True, 
                    d=1, D=1, error_action="warn", suppress_warnings=True, 
                    stepwise=True, random_state=20, n_fits=30, n_steps=10)

  #print(model.summary())
  model.fit(train_u)

  forecast = model.predict(n_periods=len(valid_u))

  MSES += np.sum((forecast - valid_u)**2)
 
   #mse, mse_toto = mse_measure(forecast, valid_u)


# In[ ]:


print('Total RMSE is: ', MSES/(len(train)*10))

