import numpy as np
from numpy import ma
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from matplotlib import rc
import FFfromLCSM, os, json
from pprint import pprint


with open('corr.json') as data_file:
    corr = json.load(data_file)


corr_matrix = np.zeros((21,21))
blocks = [] #list of all the 49 blocks 

#loop over the dictionary keys to estrapolate single 3x3 blocks
for FFcomb in corr['correlation']:
    block = np.array([])
    for element in corr['correlation'][FFcomb]:
        block = np.append(block, corr['correlation'][FFcomb][element])
    block = block.reshape(3,3)        
    blocks.append(block)

#the list is converted to an array
block_array = np.asarray(blocks)

#generate a 21x21 matrix looping over
#all the elements of the block_array
for n in range(len(block_array)):
    row = n // 7
    col = n % 7
    for i in range(3):
        for j in range(3):
            corr_matrix[3*row+i,3*col+j] = block_array[n,i,j]

#some off-diagonal elements are 1
#this will set them to 0.99
for i in range(len(corr_matrix)):
    for j in range(len(corr_matrix)):
        if i != j:
            if corr_matrix[i,j] == 1:
                corr_matrix[i,j] = 0.99
                

sigma_array = np.array([])

for FF in corr['uncertainty']:
    for element in corr['uncertainty'][FF]:
        sigma_array = np.append(sigma_array, corr['uncertainty'][FF][element])

sigma_t = sigma_array[:, np.newaxis]
sigma_matrix = sigma_t * sigma_array


variance_matrix = np.zeros((21,21))

for i in range(len(sigma_matrix)):
    for j in range(len(sigma_matrix)):
        variance_matrix[i,j] = corr_matrix[i,j] * sigma_matrix[i,j]
        

q = np.arange(0, 20, 0.5)

#n specify which FF we want
#0 = A0, 1=A1 ...
def DeltaFF(q, n):
    deltaFF = np.zeros(len(q))
    for z in range(len(q)):
        res = np.dot(variance_matrix[3*n : 3*n+3, 3*n : 3*n+3],\
          FFfromLCSM.DcoeffFF(q)[z, n, :])
        res1 = np.dot(np.transpose(FFfromLCSM.DcoeffFF(q)[z, n, :]), res)
        deltaFF[z] =  np.sqrt(res1)
    return(deltaFF)


def FFplusDFF(q, n):
    return(DeltaFF(q, n) + FFfromLCSM.FF(q)[:, n])


def FFminusDFF(q, n):
    return(FFfromLCSM.FF(q)[:, n] - DeltaFF(q, n))


rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
plt.plot(q, FFplusDFF(q, 0))
plt.plot(q, FFminusDFF(q, 0))
plt.xlabel('$q^2$')
plt.ylabel('$A_{0} (q^2)$')
plt.show()


    
