"""Calculates and plots autocorrelation function against algorithm moves"""

import numpy as np
from time import time
rng = np.random.default_rng()
import matplotlib.pylab as plt
import functions.initial as initial
import functions.metropolis as metropolis
import functions.cluster as cluster

width = 40  # width of the lattice
betaJ = 0.4407

# Metropolis-Hastings ACF
mag_MH = []
MH_n_steps = 2000  # number of steps to run each simulation for MH
MH_burn_in = 10000  # equilibration time for MH

lattice = initial.create_lattice(width, 0)
metropolis.n_MH_moves(lattice, width, betaJ, MH_burn_in) # equilibration

for step in range(MH_n_steps):
    metropolis.MH_flip(lattice, width, betaJ)
    mag_MH.append(initial.magnetisation(lattice))
    if step%100 == 0:
        print(step)

#Wolf ACF
p_add = 1 - np.exp(-2*betaJ)
mag_wolff = []
wolff_n_steps = 2000 # number of moves for wolff algorithm
wolff_burn_in = 10000 # equilibration time for wolff

lattice = initial.create_lattice(width, 0)
cluster.n_wolff_moves(lattice, p_add, wolff_burn_in) # equilibration

for step in range(wolff_n_steps):
    cluster.wolff_flip1(lattice, p_add)
    mag_wolff.append(initial.magnetisation(lattice))
    if step%100 == 0:
        print(step)

start1 = time()
ACF_MH = initial.autocorrelation(mag_MH)
end1 = time()
sm_time1 = end1-start1
start2 = time()
ACF_wolff = initial.autocorrelation(mag_wolff)
end2 = time()
sm_time2 = end2-start2
start3 = time()
ACF_MH_2 = initial.autocorrelation_2(mag_MH,len(mag_MH))
end3 = time()
basic_time1 = end3-start3
start4 = time()
ACF_wolff_2 = initial.autocorrelation_2(mag_wolff, len(mag_wolff))
end4 = time()
basic_time2 = end4-start4

print('statsmodels acf runtime = ', sm_time1, 'and', sm_time2)
print('basic acf runtime = ', basic_time1, 'and', basic_time2)

data = [ACF_MH,ACF_wolff]
np.save('acf_comp3', data)
data2 = [ACF_MH_2,ACF_wolff_2]
np.save('acf_basic3', data2)


plt.figure()
plt.plot(ACF_wolff, label='Wolff', color='blue')
plt.plot(ACF_MH, label='Metropolis-Hastings', color='red')
plt.plot(ACF_wolff_2, label='Wolff2', color='green')
plt.plot(ACF_MH_2, label='Metropolis-Hastings2', color='orange')
plt.xlabel('Lag time')
plt.ylabel('ACF')
plt.title(r'ACF for MH vs Wolff at critical temperature')
plt.legend()
plt.show()