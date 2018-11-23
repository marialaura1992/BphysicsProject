import numpy as np
from numpy import ma
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from matplotlib import rc
from pypet import Environment, cartesian_product
import FFfromLCSM, os

#Range of q^2 values

q = np.arange(0, 20, 0.5)


#Coefficients uncertainties

Uncertainties = np.array([
    [0.03, 0.26, 1.63],  #A0
    [0.03, 0.19, 1.03],  #A1
    [0.02, 0.13, 0.66],  #A12
    [0.03, 0.26, 1.53],  #V
    [0.03, 0.19, 1.64],  #T1
    [0.03, 0.17, 0.80],  #T2
    [0.06, 0.22, 2.20]]) #T23


#Coefficients central values

CentralCoeff = np.array([
    [0.37, -1.37, 0.13], #A0
    [0.30, 0.39, 1.19],  #A1
    [0.27, 0.53, 0.48],  #A12
    [0.38, -1.17, 2.42], #V
    [0.31, -1.01, 1.53], #T1
    [0.31, 0.50, 1.61],  #T2
    [0.67, 1.32, 3.82]]) #T23

#Loop over all the FFs, to estimate the max and min
#values. No correlation between coefficients a_1, a_2, a_3.
#Simple parameter space scan.

for i in range(len(Uncertainties)):
    MaxCoeff = CentralCoeff[i] + Uncertainties[i]
    MinCoeff = CentralCoeff[i] -  Uncertainties[i]

    FFcentralEval = FFfromLCSM.FF(q, CentralCoeff)

    
    # Create an environment
    env = Environment(overwrite_file=True)
    # Get the trajectory from the environment
    traj = env.traj
    # Add parameters
    traj.f_add_parameter('a1',  np.float64(1), comment = 'First dimension')
    traj.f_add_parameter('a2',  np.float64(1), comment = 'Second dimension')
    traj.f_add_parameter('a3',  np.float64(1), comment = 'Third dimension')
    
    traj.f_explore(cartesian_product ({'a1' : [MinCoeff[0], CentralCoeff[i][0], MaxCoeff[0]],
                                       'a2' : [MinCoeff[1], CentralCoeff[i][1], MaxCoeff[1]],
                                       'a3' : [MinCoeff[2], CentralCoeff[i][2], MaxCoeff[2]]
    }))
    
    
    # Define the observable in the par. space
    def scan(traj):
        CentralCoeff[i][0] = traj.a1
        CentralCoeff[i][1] = traj.a2
        CentralCoeff[i][2] = traj.a3
        return (FFfromLCSM.FF(q, CentralCoeff))

    Result = env.run(scan)
    
    #Results is a sequence of lists, we want to put in a nicer form,
    #tensor of dimension (N^ParameterSpacePoints, N^qvalues, N^FFs) 
    Results = np.zeros((len(Result), len(q), len(Uncertainties)))
   
    for l in range(len(Result)):
        for j in range(len(q)):
            for k in range(len(Uncertainties)):
                Results[l][j][k] = Result[l][1][j][k]

    #Find for each q the max and min value.
    MaxValue = np.zeros(len(q))
    MinValue = np.zeros(len(q))
    for j in range(len(q)):
        MaxValue[j] = max(Results[:,j,i])
        MinValue[j] = min(Results[:,j,i])
        

    #What we want to plot.    
    Fmax = MaxValue[:]
    Fmin = MinValue[:]
    Fcentral = FFcentralEval[:,i]
    
    #Make the plot.
    rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)
    plt.plot(q, Fmax, 'b--')
    plt.plot(q, Fmin, 'b--')
    plt.plot(q, Fcentral, 'b',  label = 'LCSM + Lattice' )
    plt.fill_between(q, Fmax , Fmin, color='blue', alpha='0.3')
    FFnames = np.array(['A_0', 'A_1', 'A_12', 'V', 'T_1', 'T_2', 'T_23' ])
    plt.xlabel('$q^2$')
    plt.ylabel('${}(q^2)$'.format(FFnames[i]))
    plt.legend()
    plt.gca().set_ylim([0, 2])
    pathname = 'Figure/{}.png'.format(FFnames[i])
    if os.path.isfile(pathname):
        os.remove(pathname)
    plt.savefig(pathname, bbox_inches='tight')
    plt.clf()
   
