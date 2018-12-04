import numpy as np
from numpy import ma
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from matplotlib import rc
import FFfromLCSM, os, json
from pprint import pprint

Central_Coefficients = np.array([
    [0.37, -1.37, 0.13], #A0
    [0.30, 0.39, 1.19],  #A1
    [0.27, 0.53, 0.48],  #A12
    [0.38, -1.17, 2.42], #V
    [0.31, -1.01, 1.53], #T1
    [0.31, 0.50, 1.61],  #T2
    [0.67, 1.32, 3.82]])  #T23


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
          FFfromLCSM.DcoeffFF(q, Central_Coefficients)[z, n, :])
        res1 = np.dot(np.transpose(FFfromLCSM.DcoeffFF(q, Central_Coefficients)[z, n, :]), res)
        deltaFF[z] =  np.sqrt(res1)
    return(deltaFF)


def FFplusDFF(q, n):
    return(DeltaFF(q, n) + FFfromLCSM.FF(q, Central_Coefficients)[:, n])


def FFminusDFF(q, n):
    return(FFfromLCSM.FF(q, Central_Coefficients)[:, n] - DeltaFF(q, n))



for i in range(7):
    FFmax = FFplusDFF(q, i)[:]
    FFmin = FFminusDFF(q, i)[:]
    
    rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)
    plt.plot(q, FFmax, 'b', label = 'LCSM+Lattice')
    plt.plot(q, FFmin, 'b')
    plt.fill_between(q, FFmax, FFmin, color='blue', alpha='0.3')
    FFnames = np.array(['A_0', 'A_1', 'A_12', 'V', 'T_1', 'T_2', 'T_23' ])
    plt.xlabel('$q^2$')
    plt.ylabel('${}(q^2)$'.format(FFnames[i]))
    plt.legend()
    plt.gca().set_ylim([0, 2])
    pathname = 'Figure/ErrorPropWithCorr/{}.png'.format(FFnames[i])
    if os.path.isfile(pathname):
        os.remove(pathname)
    plt.savefig(pathname, bbox_inches='tight')
    plt.clf()


    
