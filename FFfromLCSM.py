from __future__ import division 
import numpy as np
import scipy as sp
from numpy import ma
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from matplotlib import rc


#B->K* FF coefficients from
#https://arxiv.org/pdf/1503.05534.pdf pg. 12
#for each of the seven FF only 3 coeffcients
#are needed for the fit between LCSM and LQCD

Central_Coefficients = np.array([
    [0.37, -1.37, 0.13], #A0
    [0.30, 0.39, 1.19],  #A1
    [0.27, 0.53, 0.48],  #A12
    [0.38, -1.17, 2.42], #V
    [0.31, -1.01, 1.53], #T1
    [0.31, 0.50, 1.61],  #T2
    [0.67, 1.32, 3.82]])  #T23


Masses = {'m_B' : 5.27963,
          'm_K*' : 0.89555,
          'm_R' : [5.366, 5.829, 5.829, \
                   5.415, 5.415, 5.829, 5.829]
}


def t(sign):
    if sign == '+':
        res = (Masses['m_B'] + Masses['m_K*'])**2
    elif sign == '-':
        res = (Masses['m_B'] - Masses['m_K*'])**2
    return res


t0 = t('+') * (1 - np.sqrt(1 - (t('-')/t('+'))))

#q stands for q**2!!!

def P(q):
    res = np.array([])
    for i in range(len(Masses['m_R'])):
        res = np.append(res, (1 - (q/Masses['m_R'][i]**2))**-1)
    return res


def z(q):
    factor1 = np.sqrt(t('+') - q) - np.sqrt(t('+') - t0)
    factor2 = np.sqrt(t('+') - q) + np.sqrt(t('+') - t0)
    return factor1/factor2


#q = np.arange(0, 20, 0.1)

def FF(q):
    res = np.zeros((len(q), len(Central_Coefficients)))
    for k in range(len(q)):
        for i in range(len(Central_Coefficients)):
            res1 = 0
            for j in range(len(Central_Coefficients[0])):
                res1 += Central_Coefficients[i][j] * \
                        (z(q[k]) - z(0))**j
            res[k][i] = P(q[k])[i] * res1
    return res

def DcoeffFF(q):
    res = np.zeros((len(q), len(Central_Coefficients), len(Central_Coefficients[0])))
    for k in range(len(q)): 
        for i in range(len(Central_Coefficients)): #7
            for j in range(len(Central_Coefficients[0])): #3
                res[k][i][j] = (z(q[k]) - z(0))**j * P(q[k])[i]
    return res

'''
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
plt.plot(q, FF(q)[:,0], 'b--', label = 'LCSM + Lattice')
plt.xlabel('$q^2$')
plt.ylabel('$T_{12} (q^2)$')
plt.legend()
plt.gca().set_ylim([0,1.5])
plt.show()
'''
