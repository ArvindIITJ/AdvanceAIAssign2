#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt

#initialization
#measured values
measured_temp=np.random.uniform(30,35,10)
#print(measured_temp)

#actual values
actual_temp=np.full(10,32)
#print(actual_temp)

#error
error=np.zeros(10)
#print(error)

#kalman gain
kalman_gain=np.zeros(10)
#print(kalman_gain)

#estimated values
estimated_temp=np.zeros(10)
#print(estimated_temp)

#initial conditions
estimated_temp[0]=32
error[0]=measured_temp[0]-estimated_temp[0]
kalman_gain[0]=1

#print(estimated_temp[0],error[0],kalman_gain[0])

#iterations
for i in range(1,10):
    estimated_temp[i]=estimated_temp[i-1]+kalman_gain[i-1]*error[i-1]
    error[i]=measured_temp[i]-estimated_temp[i]
    kalman_gain[i]=kalman_gain[i-1]*(1-kalman_gain[i-1]*error[i-1])

print(estimated_temp)

#plotting
t=np.arange(0,10,1)
plt.plot(t,actual_temp,label="actual temperature")
plt.plot(t,measured_temp,label="measured temperature")
plt.plot(t,estimated_temp,label="estimated temperature")
plt.xlabel("time")
plt.ylabel("temperature")
plt.legend()
plt.show()

