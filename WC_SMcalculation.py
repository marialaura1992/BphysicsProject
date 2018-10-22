from __future__ import division 
import numpy as np
import scipy as sp
from numpy import ma
import matplotlib.pyplot as plt
import scipy.integrate as integrate


#Analytical expression for the WC C_1-C_10 in the decay B-> Xs l^+l^-
#Mostly taken from Buras, Muenz  arXiv:hep-ph/9501281
#The operator basis is the same as the one they use
#For a LL calculation, C_1-C_7 must be evaluated at LL but
#C_9 at NLL.


parameters = {'alpha' : 1/129,
              'sin_theta_w**2' : 0.23,
              'beta_0' : 23/3,
              'beta_1' : 116/3,
              'Lambda' : 0.225,
              'M_w' : 80,
              'm_t' : 170,
              'm_b' : 4.8,
              'm_c' : 1.4
}



#set of coefficients needed for the WC1-6  arXiv:hep-ph/9501281 pg 4

coefficients = {'a' : [14/23, 16/23, 6/23, -12/23, 0.4086,\
                       -0.4230, -0.8994, 0.1456],
                'k' : [[0, 0, 1/2, -1/2, 0, 0, 0, 0],
                       [0, 0, 1/2, 1/2, 0, 0, 0, 0],
                       [0, 0, -1/14, 1/6, 0.0510, -0.1403,\
                        -0.0113, 0.0054],
                       [0, 0, -1/14, -1/6, 0.0984, 0.1214, \
                        0.0156, 0.0026],
                       [0, 0, 0, 0, -0.0397, 0.0117, -0.0025, 0.0304],
                       [0, 0, 0, 0, 0.0335, 0.0239, -0.0462, -0.0112]],
                'h' : [2.2996, -1.0880, -3/7, -1/14, -0.6494,\
                       -0.0380, -0.0186, -0.0057],
                'p' : [0, 0, -80/203, 8/33, 0.0433,\
                       0.1384, 0.1648, -0.0073],
                'r' : [0, 0, 0.8966, -0.1960, -0.2011,\
                        0.1328, -0.0292, -0.1858],
                's' : [0, 0, -0.2009, -0.3579, 0.0490,\
                       -0.3616, -0.3554, 0.0072],
                'q' : [0, 0, 0, 0, 0.0318,\
                       0.0918, -0.2700, 0.0059]
}

######################################################################

#Definition of alpha_s(mu), at LL and NLL order.

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



#####################################################################

#Function needed for the loop integrals, x is the ratio of the mass
#  of the quark running into the loop and M_W.



x =  parameters['m_t']**2/parameters['M_w']**2


A =  x * (8 * x**2 + 5 * x -7)/(12 * ((x-1)**3)) +\
     x**2 * (2 - 3 * x) * np.log(x)/(2 * (x-1)**4)




B = x/(4 * (1-x)) +\
    x * np.log(x)/(4 * ((x - 1)**2))



C =  x * (x - 6)/(8 * (x-1)) +\
     x * (3 * x + 2) * np.log(x)/(8 * ((x - 1)**2))



D =  (-19 * x**3 + 25 * x**2)/\
     (36 * (x - 1)**3) + (x **2 * (5 * x**2 - 2 * x - 6)) *\
     np.log(x)/(18 * (x - 1)**4) - 4/9 * np.log(x)



E = x * (18 - 11 * x - x **2)/\
    (12 * (1 - x)**3) + (x **2 * (15 - 16 * x + 4 * x**2)) *\
    np.log(x)/ (6 * (1 - x)**4) - 2/3 * np.log(x)



F = x * (x**2 - 5 * x -2)/(4 * (x - 1)**3) +\
    3 * x**2 * np.log(x)/(2 * (x - 1)**4)



C_7 =  -1/2 * A



C_8 = -1/2 * F

##################################################################


#Functions needed for WC9.

def P_0(mu):
    factor1 = 0
    for i in range(len(coefficients)):
        factor1 += coefficients['p'][i] * \
                   (eta(mu, 'NLL')**(coefficients['a'][i]+1))
    factor2 = np.pi/(alpha_s(parameters['M_w'], 'NLL')) *\
              (-0.1875 + factor1)
    factor3 = 0
    for i in range(len(coefficients)):
        factor3 += (coefficients['r'][i] + \
                   coefficients['s'][i] * eta(mu, 'NLL')) *\
                   eta(mu, 'NLL')**coefficients['a'][i]
    return factor2 + 1.2468 + factor3



def P_E(mu):
    factor = 0
    for i in range(len(coefficients)):
        factor += coefficients['q'][i] * \
                   eta(mu, 'NLL')**(coefficients['a'][i]+1)
    return 0.1405 + factor

#################################################################

#Functions needed for WC9_eff
def s_hat(q): #q is the lepton pair invariant mass squared.
    return q/parameters['m_b']**2


z = parameters['m_c']/parameters['m_b']


def h(mu, z, q):
    if z == 0:
        factor = 8/27 - 8/9 * np.log(parameters['m_b']/mu) -\
                 4/9 * np.log(s_hat(q)) + 4j/9 * np.pi
        return factor
    else:
        w = 4 * z**2/s_hat(q)
        factor1 = -8/9 * np.log(parameters['m_b']/mu) - 8/9 * np.log(z) +\
                  8/27 + 4/9 * w
        if w < 1:
            factor2 = - 2/9 * (2 + w) * np.absolute(1 - w)**1/2 *\
                      np.log(np.absolute(np.sqrt(1 - w) + 1)/(np.sqrt(1 - w) - 1)) -\
                      jnp.pi
        elif w > 1:
            factor2 = - 2/9 * (2 + w) * np.absolute(1 - w)**1/2 *\
                      2 * np.arctan(1/(np.sqrt(w - 1)))
        return factor1 + factor2

    
# sp.special.spence(x) gives Li2(1 - x),
# we want Li2(x), so we call sp.special.spence(1-x).   
def Li2(q): 
    return sp.special.spence(1 - s_hat(q))


def omega(q):
    factor = -2/9 * np.pi**2 - 4/3 * Li2(q) -\
             2/3 * np.log(s_hat(q)) * np.log(1- s_hat(q))-\
             (5 + 4 * s_hat(q))/(3 * (1 + 2 * s_hat(q))) *\
             np.log(1 - s_hat(q)) -\
             (2 * s_hat(q) * (1 + s_hat(q)) * (1 - 2 * s_hat(q)))/\
             (3 * (1 - s_hat(q))**2 * (1 + 2 * s_hat(q))) *\
             np.log(s_hat(q)) + (5 + 9 * s_hat(q) - 6 * s_hat(q)**2)/\
             (6 * (1 - s_hat(q)) * (1 + 2 * s_hat(q)))
    return factor



def eta_tilde(mu, q):
    return 1 + (alpha_s(mu, 'NLL')/np.pi) * omega(q) 


#########################################################################
# Definiiton of the WCs arXiv:hep-ph/9501281.
# WC1-7 are defined at LL order.
# WC8_eff doesn't enter at this level of accuracy rXiv:hep-ph/9501281 pg. 4
# WC9 must include NLL order corrections arXiv:hep-ph/9501281 pg. 7 and
# arXiv:hep-ph/9806471 pg. 124.
# WC10 doesn't renormalize, no mu dependece a part from the top mass in the
# loop functions B and C. 


def WC1_6(mu): #returns a list with WC1 - WC6 
    WC_coefficients = [0]*6
    for j in range(len(WC_coefficients)):
        for i in range(len(coefficients)):
            WC_coefficients[j] += coefficients['k'][j][i] *\
                     eta(mu, 'LL')**(coefficients['a'][i])
    return WC_coefficients



def WC7_eff(mu):
    factor1 = eta(mu, 'LL')**(16/23) * C_7 + 8/3 *\
              (eta(mu, 'LL')**(14/23)- eta(mu, 'LL')**(16/24)) * C_8
    factor2 = 0
    for i in range(len(coefficients)):
        factor2 += coefficients['h'][i] *\
                   eta(mu, 'LL')**(coefficients['a'][i])
    return factor1 + factor2



WC10 = parameters['alpha']/(2 * np.pi) *\
       (B - C)/parameters['sin_theta_w**2']



def WC9(mu):
   return  P_0(mu) + (C - B)/parameters['sin_theta_w**2'] -\
       4 * (C + 1/4 * D) + P_E(mu) * E



def C_9(mu):
    return parameters['alpha']/(2 * np.pi) * WC9(mu)



def WC9_eff(mu, q):
    factor1 = WC9(mu) * eta_tilde(mu, q)
    factor2 = h(mu, z, q) * ( 3 * WC1_6(mu)[0] +\
                                    WC1_6(mu)[1] +\
                       3 * WC1_6(mu)[2] + WC1_6(mu)[3] +\
                       3 * WC1_6(mu)[4] + WC1_6(mu)[5] )
    factor3 = -1/2 * h(mu, 1, q) * (4 * WC1_6(mu)[2] +\
                              4 * WC1_6(mu)[3] +\
                              3 * WC1_6(mu)[4] +\
                              WC1_6(mu)[5])
    factor4 = -1/2 * h(mu, 0, q) * ( WC1_6(mu)[2] +\
                        3 * WC1_6(mu)[3]) +\
                        2/9 * (3 * WC1_6(mu)[2] +\
                        WC1_6(mu)[3] + 3 * WC1_6(mu)[4] +\
                               WC1_6(mu)[5])
    return factor1 + factor2 + factor3 + factor4


