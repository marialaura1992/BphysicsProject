from __future__ import division 
import numpy as np
from numpy import ma
import matplotlib.pyplot as plt


#Analytical expression for the WC C_1-C_10 in the decay B-> Xs l^+l^-
#Mostly taken from Buras, Muenz  arXiv:hep-ph/9501281
#The operator basis is the same as the one they use
#The calculation is at LL for C_1-C_6 and NLL for C_9


parameters = {'alpha' : 1/129,
              'sin_theta_w**2' : 0.23,
              'beta_0' : 23/3,
              'beta_1' : 116/3,
              'Lambda' : 0.225,
              'alpha_s_Mz' : 0.118,  # \pm 0.005
              'M_w' : 80,
              'm_t' : 170,
              'm_b' : 4.8,
              'm_c' : 1.4}



#set of coefficients needed for the WC1-6  arXiv:hep-ph/9501281 pg 4
coefficients = {'a' : [14/23, 16/23, 6/23, -12/23, 0.4086, -0.4230, -0.8994, 0.1456],
                'k' : [[0, 0, 1/2, -1/2, 0, 0, 0, 0],
                       [0, 0, 1/2, 1/2, 0, 0, 0, 0],
                       [0, 0, -1/14, 1/6, 0.0510, -0.1403, -0.0113, 0.0054],
                       [0, 0, -1/14, -1/6, 0.0984, 0.1214, 0.0156, 0.0026],
                       [0, 0, 0, 0, -0.0397, 0.0117, -0.0025, 0.0304],
                       [0, 0, 0, 0, 0.0335, 0.0239, -0.0462, -0.0112]],
                'h' : [2.2996, -1.0880, -3/7, -1/14, -0.6494, -0.0380, -0.0186, -0.0057]
}



def alpha_s(scale, order):   #from Buras arXiv:hep-ph/9806471 pg.47
    if order == 'LL' :   
        c = 0
    if order == 'NLL':
        c = 1
    return  (4 * np.pi)/(parameters['beta_0'] *\
                        np.log(scale**2/parameters['Lambda']**2)) *\
                        (1- c * (parameters['beta_1'] *\
                             np.log(np.log(scale**2/parameters['Lambda']**2)))/\
                         (parameters['beta_0']**2 *\
                          np.log(scale**2/parameters['Lambda']**2)))
  
   

def eta(mu, order):
    return alpha_s(parameters['M_w'], order)/alpha_s(mu, order)



def A(x):
    return  x * (8 * x**2 + 5 * x -7)/(12 * (x-1)**3) +\
                   x**2 * (2- 3 * x) * np.log(x)/(2 * (x-1)**4)


def F(x):
    return  x * (x**2 - 5 * x -2)/(4 * (x-1)**3) +\
                  3 * x**2 * np.log(x)/(2 * (x-1)**4)
    

def C_7(x):
    return -1/2 * A(x)



def C_8(x):
    return -1/2 * F(x)




def WC1_6(mu, order): #For WC=1...6 arXiv:hep-ph/9501281 eq.  2.2
    WC_coefficients = [0]*6
    for j in range(len(WC_coefficients)):
        for i in range(len(coefficients)):
            WC_coefficients[j] += coefficients['k'][j][i] *\
                     eta(mu, order)**(coefficients['a'][i])
    return WC_coefficients


def WC7_eff(mu, order, x):
    factor1 = eta(mu, order)**(16/23) * C_7(x) + 8/3 *\
              (eta(mu, order)**(14/23)- eta(mu, order)**(16/24)) * C_8(x)
    factor2 = 0
    for i in range(len(coefficients)):
        factor2 += coefficients['h'][i] * eta(mu, order)**(coefficients['a'][i])
    return factor1 + factor2
    

x =  parameters['m_t']**2/parameters['M_w']**2
print WC7_eff(4.2, 'LL', x)
