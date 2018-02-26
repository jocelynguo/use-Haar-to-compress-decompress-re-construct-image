# use-Haar-to-compress-decompress-re-construct-image
Compress Script
February 16, 2018
In [1]: # import the packages
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
import numpy
In [2]: # define a function to covert the image to a gray scale image
def rgb2gray(rgb):
return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
# define a function to get the proper Haar matrix and permutation matrix
def GetHaarMatrices(N):
Q = np.matrix("[1,1;1,-1]")
M = int(N/2)
T = np.kron(matlib.eye(M),Q)/np.sqrt(2)
P = np.vstack((matlib.eye(N)[::2,:],matlib.eye(N)[1::2,:]))
return T,P
In [3]: # reads in a jpeg image
A = imageio.imread('image.jpg')
# show the original image just read in
plt.imshow(A, cmap = plt.get_cmap('gray'))
plt.title("original image")
plt.show()
1
In [4]: # resize the image(before apply gray scale function) as a 256 by 256 matrix
A = skimage.transform.resize(A, [256, 256], mode='constant')
#print(A)
# show the jpeg image in a figure
plt.imshow(A, cmap = plt.get_cmap('gray'))
plt.title("original image after resize")
plt.show()
2
In [5]: # Apply the rgb2gray function to the image
A = rgb2gray(A)
#print(np.amax(A))
# show the jpeg image in a figure
plt.imshow(A, cmap = plt.get_cmap('gray'))
plt.title("256 by 256 Gray-scale")
plt.show()
3
In [6]: # make a deep copy of resize&gray-scale image
B = copy.deepcopy(A)
# set size to 256
N = 256
# Doing full-level Encoding (Forward Haar Transform)
for i in range(int(np.log2(N))):
T,P = GetHaarMatrices(N)
#print(T.shape)
B[0:N, 0:N] = P*T*B[0:N, 0:N]*T.T*P.T
N = int(N/2)
# show the result of full-level encoding
plt.figure()
plt.imshow(B[128:256,128:256], cmap = plt.get_cmap('gray'))
plt.title("Full-level Haar Encoding")
plt.show()
# print the info of B
print(B)
4
[[ 1.37032397e+02 8.28231535e-01 2.24150933e+00 ... 7.84313725e-03
-3.92156863e-03 -3.37009804e-04]
[ 4.66996372e+00 -4.49479731e+00 -1.76164275e+00 ... 7.84313725e-03
-3.92156863e-03 -3.37009804e-04]
[-6.79411587e+00 -1.14338176e+01 -6.27488108e-01 ... 3.92156863e-03
-3.92156863e-03 -3.37009804e-04]
...
[ 2.56199918e-02 -1.04924725e-01 -2.73541940e-02 ... 1.92135713e-02
5.18132643e-02 -1.91346419e-02]
[ 1.64313295e-01 -4.03217201e-02 -5.23184702e-02 ... 1.17415065e-02
3.46326596e-02 -2.19951143e-02]
[-6.83361694e-02 -7.27696756e-02 -1.01429493e-01 ... -4.42582886e-02
1.48929418e-01 -7.86932466e-02]]
In [7]: # create an empty numpy array record the sign of array
sign = np.empty([256,256])
# record the sign
for i in range(B.shape[0]):
for j in range(B.shape[1]):
if B[i][j]<=0:
sign[i][j] = -1
else:
sign[i][j] = 1
5
print(sign)
[[ 1. 1. 1. ... 1. -1. -1.]
[ 1. -1. -1. ... 1. -1. -1.]
[-1. -1. -1. ... 1. -1. -1.]
...
[ 1. -1. -1. ... 1. 1. -1.]
[ 1. -1. -1. ... 1. 1. -1.]
[-1. -1. -1. ... -1. 1. -1.]]
In [8]: # make 2 deep copy of B
X = abs(copy.deepcopy(B))
Y = copy.deepcopy(B)
# convert X(2D numpy array) into 1D numpy array
Y = Y.ravel()
print(X)
[[1.37032397e+02 8.28231535e-01 2.24150933e+00 ... 7.84313725e-03
3.92156863e-03 3.37009804e-04]
[4.66996372e+00 4.49479731e+00 1.76164275e+00 ... 7.84313725e-03
3.92156863e-03 3.37009804e-04]
[6.79411587e+00 1.14338176e+01 6.27488108e-01 ... 3.92156863e-03
3.92156863e-03 3.37009804e-04]
...
[2.56199918e-02 1.04924725e-01 2.73541940e-02 ... 1.92135713e-02
5.18132643e-02 1.91346419e-02]
[1.64313295e-01 4.03217201e-02 5.23184702e-02 ... 1.17415065e-02
3.46326596e-02 2.19951143e-02]
[6.83361694e-02 7.27696756e-02 1.01429493e-01 ... 4.42582886e-02
1.48929418e-01 7.86932466e-02]]
In [9]: # make a deep copy to X to get the threshold but not affect X
Z = copy.deepcopy(Y)
# sort the numpy array by its absolute value
Z = np.sort(abs(Z))
# promopt to ask user what the top percent pixel will retain the same
cutoff = input('How many percents of smallest elements you want to set to zero?')
# define thresholding function to find the threshold
def find_th(source, percentage):
index = 0
index = math.floor(len(source) * percentage / 100)
6
threshold = source[index]
return threshold
How many percents of smallest elements you want to set to zero?95
In [10]: # apply the thresholding function to find the threshold th
th = find_th(Z, int(cutoff))
# print(th)
# implementation of the threshold process to numpy array X
for i in range(X.shape[0]):
for j in range(X.shape[1]):
if X[i][j] > th:
continue
else:
X[i][j] = 0
# show the image after apply to threshold
plt.imshow(X[128:256,128:256], cmap = plt.get_cmap('gray'))
plt.title("After Thresholding")
plt.show()
# print the matrix out the make sure A apply to the threshold function correctly
print(X)
7
[[137.032397 0.82823154 2.24150933 ... 0. 0.
0. ]
[ 4.66996372 4.49479731 1.76164275 ... 0. 0.
0. ]
[ 6.79411587 11.43381759 0.62748811 ... 0. 0.
0. ]
...
[ 0. 0. 0. ... 0. 0.
0. ]
[ 0.1643133 0. 0. ... 0. 0.
0. ]
[ 0. 0. 0. ... 0. 0.
0. ]]
In [11]: # initialize the value to create proper partition and codebook
MX = np.amax(X)
bits = int(math.log2(X.shape[0]))
NP = 2**(bits-1)-1
c1 = 0
diff = (MX/th)**(1/NP)
# create empty list of partition and codebook
partition = []
codebook = [c1]
In [12]: # create partition list
for n in range(NP):
partition.append(th*(diff**n))
# print the length of partition list
print(len(partition))
127
In [13]: # create codebook list
for n in range(NP-1):
codebook.append(random.uniform(partition[n], partition[n+1]))
codebook.append(random.uniform(partition[len(partition)-1], partition[len(partition)-1]+# print the length of the codebook list
print(len(codebook))
128
8
In [14]: # convert M(2D numpy array) into 1D list as signal
signal = []
for i in range(X.shape[0]):
for j in range(X.shape[1]):
signal.append(X[i][j])
In [15]: # define a function to do quantization
def quantiz(signal, partition, codebook):
indices = []
quanta = []
for data in signal:
index = 0
while index<len(partition) and data>partition[index]:
index += 1
indices.append(index)
quanta.append(codebook[index])
return indices, quanta
# call the quantiz function to get indices and quantized signal list
indices, quanta = quantiz(signal, partition, codebook)
# reshape quantized signal into 2D array
quanta = np.reshape(quanta, (256,256))
print(quanta)
[[130.59818677 0.79415368 2.19811157 ... 0. 0.
0. ]
[ 4.63455098 4.45363012 1.76914495 ... 0. 0.
0. ]
[ 6.71842264 11.68084684 0.62617807 ... 0. 0.
0. ]
...
[ 0. 0. 0. ... 0. 0.
0. ]
[ 0.16562914 0. 0. ... 0. 0.
0. ]
[ 0. 0. 0. ... 0. 0.
0. ]]
In [16]: # reshape the indices into 2D array
indices = np.reshape(indices, (256,256))
print(indices)
print(type(indices))
9
[[127 31 50 ... 0 0 0]
[ 64 63 46 ... 0 0 0]
[ 71 81 26 ... 0 0 0]
...
[ 0 0 0 ... 0 0 0]
[ 1 0 0 ... 0 0 0]
[ 0 0 0 ... 0 0 0]]
<class 'numpy.ndarray'>
In [17]: # make a deep copy of image after threshholding
M = copy.deepcopy(quanta)
def log_quantiz(inp):
for i in range(inp.shape[0]):
for j in range(inp.shape[1]):
if inp[i][j] == 0:
continue
else:
inp[i][j] = math.log10(inp[i][j])
log_quantiz(M)
# show the image after apply to log quantization
plt.spy(M)
plt.show()
print(M)
10
[[ 2.11593715 -0.10009545 0.34204973 ... 0. 0.
0. ]
[ 0.66600766 0.64871415 0.24776342 ... 0. 0.
0. ]
[ 0.82726732 1.06747433 -0.20330214 ... 0. 0.
0. ]
...
[ 0. 0. 0. ... 0. 0.
0. ]
[-0.78086325 0. 0. ... 0. 0.
0. ]
[ 0. 0. 0. ... 0. 0.
0. ]]
In [18]: # make a copy of image after thresholding and log_quantiz as N
N = copy.deepcopy(indices)
# start of the lossless compression by using zlib
compressed_data = zlib.compress(N, 9)
#print(compressed_data)
compress_ratio = float(sys.getsizeof(compressed_data))/sys.getsizeof(N)
# print out the percent of lossless compression
print("compress_ratio:", compress_ratio * 100, "%")
compress_ratio: 1.492372234935164 %
In [19]: # create a file to save the compressed data
f = gzip.open('compressed_data.txt.gz', 'wb')
f.write(compressed_data)
f.close()
In [20]: # create a file to save the sign np array
f2 = open('sign.txt', 'wb+')
f2.write(sign)
f2.close()
In [21]: # create a file to save the codebook list
f1 = open('codebook.txt', 'w')
for item in codebook:
f1.write("%s\n" % item)
f1.close()
11

Uncompress Script
February 16, 2018
In [1]: # import the packages
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
In [2]: # define a function to covert the image to a gray scale image
def rgb2gray(rgb):
return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
# define a function to get the proper Haar matrix and permutation matrix
def GetHaarMatrices(N):
Q = np.matrix("[1,1;1,-1]")
M = int(N/2)
T = np.kron(matlib.eye(M),Q)/np.sqrt(2)
P = np.vstack((matlib.eye(N)[::2,:],matlib.eye(N)[1::2,:]))
return T,P
In [3]: # use zlib to uncompress the compressed_data
compressed_data = gzip.open('compressed_data.txt.gz', 'rb').read()
decompress_data = zlib.decompress(compressed_data)
# convert the byte-like object to numpy array
decompress_data = np.frombuffer(decompress_data, dtype=int)
1
# reshape the data to 2D
indices = np.reshape(decompress_data, (256, 256))
# show the image before reverse log quantization
plt.spy(indices)
plt.show()
print(indices)
[[127 31 50 ... 0 0 0]
[ 64 63 46 ... 0 0 0]
[ 71 81 26 ... 0 0 0]
...
[ 0 0 0 ... 0 0 0]
[ 1 0 0 ... 0 0 0]
[ 0 0 0 ... 0 0 0]]
In [4]: # make the codebook.txt readable for python
codebook = []
with open('codebook.txt', 'r') as f1:
codebook = [line.strip() for line in f1]
codebook = [float(i) for i in codebook]
2
print(codebook)
[0.0, 0.16562914134646228, 0.17135073430213965, 0.17896790245200203, 0.19287271932004935, 0.20633271458047975, In [5]: # using codebook and indices to recover the quanta data
quanta = np.empty((256, 256))
for i in range(quanta.shape[0]):
for j in range(quanta.shape[1]):
quanta[i][j] = codebook[indices[i][j]]
print(quanta)
[[130.59818677 0.79415368 2.19811157 ... 0. 0.
0. ]
[ 4.63455098 4.45363012 1.76914495 ... 0. 0.
0. ]
[ 6.71842264 11.68084684 0.62617807 ... 0. 0.
0. ]
...
[ 0. 0. 0. ... 0. 0.
0. ]
[ 0.16562914 0. 0. ... 0. 0.
0. ]
[ 0. 0. 0. ... 0. 0.
0. ]]
In [6]: # reverse threshold to F
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
3
# Apply reverse thresholding function to M
reverse_thresholding(G)
# show the image after apply to reverse threshold
plt.imshow(G[128:256,128:256], cmap = plt.get_cmap('gray'))
plt.title("Reverse of Thresholding")
plt.show()
print(G)
[[130.59818677 0.79415368 2.19811157 ... 0. 0.
0. ]
[ 4.63455098 4.45363012 1.76914495 ... 0. 0.
0. ]
[ 6.71842264 11.68084684 0.62617807 ... 0. 0.
0. ]
...
[ 0. 0. 0. ... 0. 0.
0. ]
[ 0.16562914 0. 0. ... 0. 0.
0. ]
[ 0. 0. 0. ... 0. 0.
0. ]]
4
In [7]: # open the sign.txt to put back the sign
sign = open('sign.txt', 'rb').read()
# convert the byte-like object to numpy array
sign = np.frombuffer(sign)
# reshape the sign to 2D numpy array
sign = np.reshape(sign, (256, 256))
print(sign)
[[ 1. 1. 1. ... 1. -1. -1.]
[ 1. -1. -1. ... 1. -1. -1.]
[-1. -1. -1. ... 1. -1. -1.]
...
[ 1. -1. -1. ... 1. 1. -1.]
[ 1. -1. -1. ... 1. 1. -1.]
[-1. -1. -1. ... -1. 1. -1.]]
In [8]: # put the nagative sign back to the correct position
G = G * sign
print(G)
[[130.59818677 0.79415368 2.19811157 ... 0. -0.
-0. ]
[ 4.63455098 -4.45363012 -1.76914495 ... 0. -0.
-0. ]
[ -6.71842264 -11.68084684 -0.62617807 ... 0. -0.
-0. ]
...
[ 0. -0. -0. ... 0. 0.
-0. ]
[ 0.16562914 -0. -0. ... 0. 0.
-0. ]
[ -0. -0. -0. ... -0. 0.
-0. ]]
In [9]: # make a deep copy of G
J = copy.deepcopy(G)
# get number of times of decoding and the starting point
N = len(J)
times = int(np.log2(N))
start = 2
# Doing full-level decoding (Backward Haar Transform)
for i in range(times):
T,P = GetHaarMatrices(start)
5
J[0:start, 0:start] = T.T*P.T*J[0:start, 0:start]*P*T
start = 2 * start
# show the result of full-level decoding
plt.figure()
plt.imshow(J, cmap = plt.get_cmap('gray'))
plt.show()
# print the info of J
print(J)
[[0.31332279 0.31332279 0.43884974 ... 0.31523406 0.31523406 0.31523406]
[0.31332279 0.31332279 0.43884974 ... 0.31523406 0.31523406 0.31523406]
[0.31332279 0.31332279 0.43884974 ... 0.31523406 0.31523406 0.31523406]
...
[0.17002039 0.17002039 0.25283496 ... 0.28278207 0.1971067 0.1971067 ]
[0.25283496 0.25283496 0.25283496 ... 0.28278207 0.1971067 0.1971067 ]
[0.25283496 0.25283496 0.25283496 ... 0.28278207 0.1971067 0.1971067 ]]
In [10]: #####################
# get PSNR#
#####################
# get the information from the original image(before full-level encoding)
# reads in the original image
6
A = imageio.imread('image.jpg')
# resize the image(before apply gray scale function) as a 256 by 256 matrix
A = skimage.transform.resize(A, [256, 256], mode='constant')
# Apply the rgb2gray function to the image
A = rgb2gray(A)
print(A)
[[0.31905515 0.3648937 0.42410076 ... 0.35208627 0.35567083 0.35600784]
[0.31661375 0.36307557 0.43115786 ... 0.35208627 0.35567083 0.35600784]
[0.31623509 0.3615686 0.43780567 ... 0.35208627 0.35567083 0.35600784]
...
[0.23155268 0.1562254 0.29268602 ... 0.31176704 0.14906503 0.21840392]
[0.23837193 0.19653385 0.30720091 ... 0.28660407 0.13619611 0.25433132]
[0.31247219 0.25910593 0.29068716 ... 0.39854037 0.252657 0.21340572]]
In [11]: # get the maximum value of the original image
maxValue = np.amax(A)
print(maxValue)
0.9999999999999999
In [12]: # get the 2D info of origianl image
print(A)
[[0.31905515 0.3648937 0.42410076 ... 0.35208627 0.35567083 0.35600784]
[0.31661375 0.36307557 0.43115786 ... 0.35208627 0.35567083 0.35600784]
[0.31623509 0.3615686 0.43780567 ... 0.35208627 0.35567083 0.35600784]
...
[0.23155268 0.1562254 0.29268602 ... 0.31176704 0.14906503 0.21840392]
[0.23837193 0.19653385 0.30720091 ... 0.28660407 0.13619611 0.25433132]
[0.31247219 0.25910593 0.29068716 ... 0.39854037 0.252657 0.21340572]]
In [13]: # get the 2D info of the reconstructed image
print(J)
[[0.31332279 0.31332279 0.43884974 ... 0.31523406 0.31523406 0.31523406]
[0.31332279 0.31332279 0.43884974 ... 0.31523406 0.31523406 0.31523406]
[0.31332279 0.31332279 0.43884974 ... 0.31523406 0.31523406 0.31523406]
...
[0.17002039 0.17002039 0.25283496 ... 0.28278207 0.1971067 0.1971067 ]
[0.25283496 0.25283496 0.25283496 ... 0.28278207 0.1971067 0.1971067 ]
[0.25283496 0.25283496 0.25283496 ... 0.28278207 0.1971067 0.1971067 ]]
7
In [14]: # calculate the MSE
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
0.0024329983400938992
In [15]: # calculate the PSNR
PSNR = 20*math.log10(maxValue) - 10*math.log10(MSE)
print(PSNR)
26.138581873652385
8
