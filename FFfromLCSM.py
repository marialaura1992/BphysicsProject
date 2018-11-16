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
    [0.27, 0.53, 0.48],  #A2
    [0.38, -1.17, 2.42], #V
    [0.31, -1.01, 1.53], #T1
    [0.31, 0.50, 1.61],  #T2
    [0.67, 1.32, 3.82]])  #T2


#Coefficients uncertainties

Delta_Coefficients = np.array([
    [0.03, 0.26, 1.63],  #A0
    [0.03, 0.19, 1.03],  #A1
    [0.02, 0.13, 0.66],  #A2
    [0.03, 0.26, 1.53],  #V
    [0.03, 0.19, 1.64],  #T1
    [0.03, 0.17, 0.80],  #T2
    [0.06, 0.22, 2.20]]) #T3



Masses = {'m_B' : 5.279,
          'm_K*' : 0.895,
          'm_R' : [5.366, 5.415, 5.829]
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
    res = [Masses['m_R'][1]]*2
    res1 = [Masses['m_R'][2]]*4
    m = [[Masses['m_R'][0]]]
    m.append(res)
    m.append(res1)
    m = [y for x in m for y in x]
    res2 = []
    for i in range(len(m)):
        res2.append((1 - (q/m[i]**2))**-1)
    return res2


def z(q):
    factor1 = np.sqrt(t('+') - q) - np.sqrt(t('+') - t0)
    factor2 = np.sqrt(t('+') - q) + np.sqrt(t('+') - t0)
    return factor1/factor2


def FF(q, value):
    if value == 'central':
        Coefficients = Central_Coefficients
    if value == 'limit':
        Coefficients = Delta_Coefficients
    res = np.array([])
    for i in range(7):
        res1 = 0
        for j in range(len(Coefficients[0])):
            res1 += Coefficients[i][j] * \
                    (z(q) - z(0))**j
        res = np.append(res, P(q)[i] * res1)
    return res


def MaxMinFF(q):
    MaxVal = FF(0, 'central') + FF(0, 'limit')
    MinVal = FF(0, 'central') - FF(0, 'limit')
    return(MaxVal, MinVal)


def FF_central(q):
    return (FF(q, 'central').tolist())


def FF_MaxMin(q):
    return (MaxMin(q).tolist())


q = np.arange(0, 20, 0.1)

rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
'''
plt.plot(q, FF(q, 'central')[0], 'b-', label = 'LCSM + Lattice')
plt.xlabel('$q^2$')
plt.ylabel('$A_0(q^2)$')
plt.legend()
plt.gca().set_ylim([0,2])
plt.show()
'''
for q in range(0,10):
    plt.plot(q, FF(q, 'central')[0])
    plt.show()
