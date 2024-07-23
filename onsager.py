"""onsager.py
Plots the analytic solution for the mean magnetisation of a 2D Ising model with no external field
"""

import numpy as np
import matplotlib.pylab as plt

T = np.linspace(0.1,4,500)
#T_crit = 2/np.log(1+np.sqrt(2))
beta = np.linspace(0,1.2,500)
beta_crit = np.log(1+np.sqrt(2))/2

def Mag_pos(x):
    if x <= beta_crit:
        return 0
    else:
        return ((1 - np.sinh(2*x)**(-4))**(1/8))
def Mag_neg(x):
    if x  <= beta_crit:
        return 0
    else:
        return -1*((1 - np.sinh(2*x)**(-4))**(1/8))
    
M_pos_b = np.zeros(len(beta))
M_neg_b = np.zeros(len(beta))

for i in range(len(beta)):
    M_pos_b[i] = Mag_pos(beta[i])
    M_neg_b[i] = Mag_neg(beta[i])

M_pos_T = np.zeros(len(T))
M_neg_T = np.zeros(len(T))

for i in range(len(T)):
    M_pos_T[i] = Mag_pos(1/T[i])
    M_neg_T[i] = Mag_neg(1/T[i])

temp_data = [T,M_pos_T]
np.save('analytical_onsager_data_temp', temp_data)

fig, ax = plt.subplots(1,2,figsize=(12,4))
plt.suptitle("Onsager solution for mean magentisation in the limit of large lattice")
ax[0].plot(T, M_pos_T, label=None, color='b')
ax[0].plot(T, M_neg_T, label=None, color='b')
ax[0].set_xlabel ('Temperature, T')
ax[0].set_ylabel ('Magnetisation, M')
ax[1].plot(beta, M_pos_b, label=None, color='r')
ax[1].plot(beta, M_neg_b, label=None, color='r')
ax[1].set_xlabel ('Inverse Temperature, Beta')
ax[1].set_ylabel ('Magnetisation, M')
plt.show()