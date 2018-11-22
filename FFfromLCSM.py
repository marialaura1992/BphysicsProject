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


#Coefficients uncertainties

Delta_Coefficients = np.array([
    [0.03, 0.26, 1.63],  #A0
    [0.03, 0.19, 1.03],  #A1
    [0.02, 0.13, 0.66],  #A12
    [0.03, 0.26, 1.53],  #V
    [0.03, 0.19, 1.64],  #T1
    [0.03, 0.17, 0.80],  #T2
    [0.06, 0.22, 2.20]]) #23



Masses = {'m_B' : 5.279,
          'm_K*' : 0.895,
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


#value can be 'central' or 'limit' while
#sign can be '-' for the central value
#'up' for positive uncertainty
#'down' for negative uncertainty

def FF(q, value, sign):
    if value == 'central' and sign == '-':
        Coefficients = Central_Coefficients
    elif value == 'limit' and sign == 'up':
        Coefficients = Central_Coefficients +\
                       Delta_Coefficients
    elif value == 'limit' and sign == 'down':
        Coefficients = Central_Coefficients -\
                       Delta_Coefficients
    else:
        return('Error')
    res = np.zeros((len(q), len(Coefficients)))
    for k in range(len(q)):
        for i in range(len(Coefficients)):
            res1 = 0
            for j in range(len(Coefficients[0])):
                res1 += Coefficients[i][j] * \
                        (z(q[k]) - z(0))**j
            res[k][i] = P(q[k])[i] * res1
    return res

q = np.arange(0, 20, 0.1)


#print('in 0 ', FF([0], 'central', '-'), '\n')

#print('in 1 ', FF([1], 'central', '-'), '\n')

#print('in 2 ', FF([2], 'central', '-'), '\n')

#print(FF(19.9, 'central', '-'), '\n')

#print(FF(q, 'central', '-'))
#print(len(FF(q, 'central', '-')))
#print(FF(q, 'central', '-').reshape(len(q), len(Central_Coefficients))[0])
ff = ['V', 'A1', 'A12', 'T1', 'T2', 'T12']


rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
plt.plot(q, FF(q, 'central', '-')[:,6], 'b--', label = 'LCSM + Lattice')
plt.xlabel('$q^2$')
plt.ylabel('$T_{12} (q^2)$')
plt.legend()
plt.gca().set_ylim([0,1.5])
plt.show()

