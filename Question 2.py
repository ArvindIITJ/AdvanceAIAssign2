#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import matplotlib.pyplot as plt 
import random 
 
# Number of jars 
J = 5 
# Number of colors 
C = 3 
# Number of observations 
N = 10 
 
# Number of balls in each jar 
balls = np.random.randint(20, 30, J) 
 
# Color distribution of the balls 
colors = np.random.randint(0, C, sum(balls)) 
 
# Initial state distribution 
pi = np.random.rand(J) 
pi = pi / sum(pi) 


# In[26]:


# Transition matrix 
A = np.random.rand(J, J) 
for i in range(J): 
    A[i] = A[i] / sum(A[i]) 
 
# Emission matrix 
B = np.random.rand(J, C) 
for i in range(J): 
    B[i] = B[i] / sum(B[i]) 
 
# Observations 
O = np.random.randint(0, C, N) 
 
# Forward algorithm 
def forward(O, pi, A, B): 
    alpha = np.zeros((N, J)) 
    alpha[0] = pi * B[:, O[0]] 
    for t in range(1, N): 
        for j in range(J): 
            alpha[t, j] = alpha[t - 1].dot(A[:, j]) * B[j, O[t]] 
    return alpha 
 
# Backward algorithm 
def backward(O, A, B): 
    beta = np.zeros((N, J)) 
    beta[N - 1] = 1
    for t in range(N - 2, -1, -1):    
         for j in range(J): 
                beta[t, j] = (beta[t + 1] * B[:, O[t + 1]]).dot(A[j, :]) 
    return beta 


# In[29]:


# Viterbi algorithm 
def viterbi(O, pi, A, B): 
    delta = np.zeros((N, J)) 
    psi = np.zeros((N, J)) 
    delta[0] = pi * B[:, O[0]] 
    for t in range(1, N): 
        for j in range(J): 
            delta[t, j] = np.max(delta[t - 1] * A[:, j]) * B[j, O[t]] 
            psi[t, j] = np.argmax(delta[t - 1] * A[:, j]) 
    return delta, psi 
 
# Posterior decoding 
def posterior_decoding(O, pi, A, B): 
    alpha = forward(O, pi, A, B) 
    beta = backward(O, A, B) 
    gamma = alpha * beta 
    for t in range(N): 
        gamma[t] = gamma[t] / sum(gamma[t]) 
    return gamma 


# In[30]:


# Baum-Welch algorithm 
def baum_welch(O, pi, A, B, iterations):  
    M = len(O) 
    N = A.shape[0] 
    xi = np.zeros((M - 1, N, N)) 
    for it in range(iterations): 
        alpha = forward(O, pi, A, B) 
        beta = backward(O, A, B) 
        for t in range(M - 1): 
            denom = np.dot(np.dot(alpha[t, :].T, A) * B[:, O[t + 1]].T, beta[t + 1, :]) 
            for i in range(N): 
                numer = alpha[t, i] * A[i, :] * B[:, O[t + 1]].T * beta[t + 1, :].T 
                xi[t, i, :] = numer / denom 
        gamma = np.sum(xi, axis=0) 
        pi = gamma[0, :] / np.sum(gamma[0, :]) 
        for i in range(N): 
            denom = np.sum(gamma[:, i]) 
            for j in range(N): 
                A[i, j] = np.sum(xi[:, i, j]) / denom 
        denom = np.sum(gamma, axis=0) 
        for k in range(C): 
            for j in range(N): 
                B[j, k] = np.sum(gamma[O == k, j]) / denom[j] 
    return pi, A, B 


# In[31]:


# Calculate P(O|λ) 
def prob(O, pi, A, B): 
    alpha = forward(O, pi, A, B) 
    return sum(alpha[N - 1]) 
 
 
 
# Print results 
print("Number of balls in each jar:") 
print(balls) 
print("Color distribution of the balls:") 
print(colors) 
print("Initial state distribution:") 
print(pi) 
print("Transition matrix:") 
print(A) 
print("Emission matrix:") 
print(B) 
print("Observations:") 
print(O) 
print("P(O|λ):") 
print(prob(O, pi, A, B))


# In[32]:


# Plot results 
plt.figure(figsize=(10, 5)) 
plt.subplot(1, 2, 1) 
plt.bar(np.arange(J), balls) 
plt.xlabel("Jar") 
plt.ylabel("Number of balls") 
plt.subplot(1, 2, 2) 
plt.bar(np.arange(C), np.bincount(colors)) 
plt.xlabel("Color") 
plt.ylabel("Number of balls") 
plt.show() 

