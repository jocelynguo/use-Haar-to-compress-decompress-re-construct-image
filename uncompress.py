
# coding: utf-8

# In[1]:


# import the packages
import numpy as np
from scipy.misc import imread, imresize, imsave
import matplotlib.pyplot as plt
from numpy import matlib
import math
from scipy import stats
import imageio
from skimage.transform import resize
import skimage
import zlib, sys
import gzip
import matplotlib
import scipy
import copy
import random
import io
import sys


# In[2]:


# define a function to covert the image to a gray scale image
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

# define a function to get the proper Haar matrix and permutation matrix
def GetHaarMatrices(N):
    Q = np.matrix("[1,1;1,-1]")
    M = int(N/2)
    T = np.kron(matlib.eye(M),Q)/np.sqrt(2)
    P = np.vstack((matlib.eye(N)[::2,:],matlib.eye(N)[1::2,:]))
    return T,P


# In[3]:


# use zlib to uncompress the compressed_data
compressed_data = gzip.open('compressed_data.txt.gz', 'rb').read()
decompress_data = zlib.decompress(compressed_data)

# convert the byte-like object to numpy array
decompress_data = np.frombuffer(decompress_data, dtype=int)

# reshape the data to 2D
indices = np.reshape(decompress_data, (256, 256))

# show the image before reverse log quantization
plt.spy(indices)
plt.show()

print(indices)


# In[4]:


# make the codebook.txt readable for python
codebook = []

with open('codebook.txt', 'r') as f1:
    codebook = [line.strip() for line in f1]
    
codebook = [float(i) for i in codebook]
    
print(codebook)


# In[5]:


# using codebook and indices to recover the quanta data
quanta = np.empty((256, 256))

for i in range(quanta.shape[0]):
    for j in range(quanta.shape[1]):
        quanta[i][j] = codebook[indices[i][j]]
        
print(quanta)


# In[6]:


# reverse threshold to F
# make a deep copy of F as G
G = copy.deepcopy(quanta)

# read in F row by row, find the min nonzero pixel
# put the number from data codebook before apply thresholding function
# in order to put the data back to nonzero
def reverse_thresholding(source):
    index = 0
    
    for i in range(source.shape[0]):
        for j in range(source.shape[1]):
            
            if source[i][j] == 0:
                index += 1
            else:
                continue
                
        
# Apply reverse thresholding function to M
reverse_thresholding(G)

# show the image after apply to reverse threshold
plt.imshow(G[128:256,128:256], cmap = plt.get_cmap('gray'))
plt.title("Reverse of Thresholding")
plt.show()

print(G)


# In[7]:


# open the sign.txt to put back the sign
sign = open('sign.txt', 'rb').read()

# convert the byte-like object to numpy array
sign = np.frombuffer(sign)

# reshape the sign to 2D numpy array
sign = np.reshape(sign, (256, 256))
print(sign)


# In[8]:


# put the nagative sign back to the correct position
G = G * sign
print(G)


# In[9]:


# make a deep copy of G
J = copy.deepcopy(G)

# get number of times of decoding and the starting point
N = len(J)
times = int(np.log2(N))
start = 2

# Doing full-level decoding (Backward Haar Transform)
for i in range(times):
    T,P = GetHaarMatrices(start)
    J[0:start, 0:start] = T.T*P.T*J[0:start, 0:start]*P*T
    start = 2 * start
    
# show the result of full-level decoding
plt.figure()
plt.imshow(J, cmap = plt.get_cmap('gray'))
plt.show()

# print the info of J
print(J)


# In[10]:


#####################
# get PSNR#
#####################

# get the information from the original image(before full-level encoding)
# reads in the original image
A = imageio.imread('image.jpg')

# resize the image(before apply gray scale function) as a 256 by 256 matrix
A = skimage.transform.resize(A, [256, 256], mode='constant')

# Apply the rgb2gray function to the image
A = rgb2gray(A)

print(A)


# In[11]:


# get the maximum value of the original image
maxValue = np.amax(A)
print(maxValue)


# In[12]:


# get the 2D info of origianl image
print(A)


# In[13]:


# get the 2D info of the reconstructed image
print(J)


# In[14]:


# calculate the MSE
MSE_arr = np.empty([J.shape[0], J.shape[1]])
for i in range(J.shape[0]):
    for j in range(J.shape[1]):
        MSE_arr[i][j] = ((A[i][j] - J[i][j])**2)
        
MSE = 0
for a in range(MSE_arr.shape[0]):
    for b in range(MSE_arr.shape[1]):
        MSE += MSE_arr[a][b]

MSE = MSE/(MSE_arr.shape[0]*MSE_arr.shape[1])

print(MSE)


# In[15]:


# calculate the PSNR
PSNR = 20*math.log10(maxValue) - 10*math.log10(MSE)
print(PSNR)

