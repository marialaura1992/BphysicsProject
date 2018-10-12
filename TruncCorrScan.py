import numpy as np
from numpy import ma
import matplotlib.pyplot as plt
from collections import OrderedDict
import matplotlib.patches as patches
import scipy.integrate as integrate
import flavio
import flavio.plots as fpl
from matplotlib import rc
from pypet import Environment, cartesian_product
import P5p_anomaly

#SMpred = P5p_anomaly.P5p_binned() #Central value prediction
#print(SMpred)


# Create an environment
env = Environment(overwrite_file=True)
# Get the trajectory from the environment
traj = env.traj
# Add parameters
traj.f_add_parameter('A0',  [1.,1., 1.] , comment = 'First dimension')
traj.f_add_parameter('A1',  [1.,1., 1.], comment = 'Second dimension')
#traj.f_add_parameter('A2bF',  1., comment = 'Second dimension')
#traj.f_add_parameter('T1bF',  1., comment = 'Third dimension')
#traj.f_add_parameter('T2bF',  1., comment = 'Fourth dimension')
#traj.f_add_parameter('T3bF',  1., comment = 'Fifth dimension')
traj.f_explore(cartesian_product ({'A0' : [ [0.002, 0.465, 1.222], [0.002, 0.715, 1.724]], #min and max values
                                   'A1' : [ [-0.038, -0.074, 0.179], [0.012, -0.038, 0.137]]
 }))

                                  # 'A2bF' : [-0.127, -0.083],
                                  # 'T1bF' : [-0.066, 0.042],
                                   #'T2bF' : [0.196, 0.11],
                                  # 'T3bF' : [0.367, 0.249]
# Define the observable in the par. space
def scan(traj):
    import imp
    imp.reload(P5p_anomaly)
    P5p_anomaly.Lmb_corr_par['A0'] = traj.A0
    P5p_anomaly.Lmb_corr_par['A1'] = traj.A1
    # P5p_anomaly.Lmb_corr_par['A2'][1]=traj.A2bF
    # P5p_anomaly.Lmb_corr_par['T1'][1]=traj.T1bF
    # P5p_anomaly.Lmb_corr_par['T2'][1]=traj.T2bF
    # P5p_anomaly.Lmb_corr_par['T3'][1]=traj.T3bF
    return P5p_anomaly.P5p_binned()


# Find the maximum and minimum value for each bin
Result = env.run(scan)
print(Result)

'''
def ManualScan():
    import imp
    P5p_anomaly.Lmb_corr_par['A0'] = [0.002, 0.465, 1.222]
    P5p_anomaly.Lmb_corr_par['A1'] = [-0.038, -0.074, 0.179]
    res = [P5p_anomaly.P5p_binned()]
    imp.reload(P5p_anomaly)
    P5p_anomaly.Lmb_corr_par['A0'] = [0.002, 0.715, 1.724]
    P5p_anomaly.Lmb_corr_par['A1'] = [0.012, -0.038, 0.137]
    res += [P5p_anomaly.P5p_binned()]
    return res

Result = ManualScan()
print(Result)
'''

def FindMaxMin():
    res = []
    Max_values = []
    Min_values = []
    for j in range(len(Result[0][1])):
        for i in range(len(Result)):
            res.append(Result[i][1][j])
        Max_values.append(max(res))
        Min_values.append(min(res))
        res=[]
    return(Max_values, Min_values)


# Find Error bars
def FindErrBar():
    MaxMin = FindMaxMin()
    bar_max = []
    bar_min = []
    for i in range(len(MaxMin[0])):
        bar_max.append( np.absolute(MaxMin[0][i] - SMpred[i]))
        bar_min.append( np.absolute(SMpred[i] - MaxMin[1][i]))
    return(bar_max, bar_min)

#for i in range(len(FindErrBar()[0])):
#    print('%i bin: ' %i, SMpred[i], '+', FindErrBar()[0][i], '-', FindErrBar()[1][i])


'''
def flattened_tensor_product(a, b):
    res = []
    for entry1 in a:
        for entry2 in b:
            res += [(entry1, entry2)]
    return res

def gen_indices(n):
    indices = [(0,), (1,)]
    for i in range(n):
        for index in indices:
            indices = flattened_tensor_product(index, (0, 1))
    return indices

'''

