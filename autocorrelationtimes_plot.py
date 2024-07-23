"""Generates data for, and plots, autocorrelation time against temperature for Wolff and MH algorithms"""

import numpy as np
rng = np.random.default_rng()
import matplotlib.pylab as plt
import functions.initial as initial
import functions.metropolis as metropolis
import functions.cluster as cluster
#import time

temps = np.arange(1.5,3,0.05)  # temperatures to simulate
width = 80 
N = width**2

# Metropolis-Hastings
print('--------MH ALGORITHM--------')
mags_MH = []
MH_n_sweeps = 100 # number of steps to run each simulation for mh
MH_burn_in = 100000  # equilibration time
for temp in temps:
    print('temp = ' + str(temp))
    betaJ = 1/temp
    lattice = initial.create_lattice(width, 1)
    ms = []
    metropolis.n_MH_moves(lattice, width, betaJ, MH_burn_in)
    
    for step in range(MH_n_sweeps):
        metropolis.n_MH_moves(lattice, width, betaJ, N)
        m = np.abs(initial.magnetisation(lattice))
        ms.append(m)

    mags_MH.append(ms)


# Wolff algorithm
print('--------WOLFF ALGORITHM--------')

mags_wolff = []
wolff_n_sweeps = 1000  # number of steps to run each sim for wolff
wolff_burn_in = 1000  # equlibration time
for temp in temps:
    print('temp = ' + str(temp))
    p_add = 1-np.exp(-2/temp)
    lattice = initial.create_lattice(width, 1)
    ms = []
    cluster.n_wolff_moves(lattice, p_add, wolff_burn_in)
    
    #for step in range(wolff_n_steps):
    #    cluster.wolff_flip1(lattice, p_add)
    #    m = np.abs(initial.magnetisation(lattice))
    #    ms.append(m)
    #    if step%10 == 0:
    #       print(step)

    num_flips = 0
    sweeps = 0
    while sweeps < wolff_n_sweeps:
        num_flips += cluster.wolff_flip1(lattice, p_add)
        sweeps += num_flips/N
        print(sweeps)
        m = np.abs(initial.magnetisation(lattice))
        ms.append(m)
    
    mags_wolff.append(ms)


#Plot and save autocorrelation time against temperature

MH_autocorr_times = []
for mags in mags_MH:
    print(len(mags))
    MH_autocorr_times.append(initial.autocorrelation_time(mags, False)/len(mags))

#save
MH_autocorr_dat = [temps,MH_autocorr_times]
np.save('MH_autocorrelation_data_4', MH_autocorr_dat)

wolff_autocorr_times = []
for mags in mags_wolff:
    print(len(mags))
    wolff_autocorr_times.append(initial.autocorrelation_time(mags, False)/len(mags))

#save
wolff_autocorr_dat = [temps,wolff_autocorr_times]
np.save('wolff_autocorrelation_data_4', wolff_autocorr_dat)

plt.plot(temps, MH_autocorr_times, label='Metropolis-Hastings')
plt.plot(temps, wolff_autocorr_times, label='Wolff')
plt.xlabel('Temperature')
plt.ylabel('Autocorrelation Time')
plt.title('Autocorrelation Time vs Temperature for each Algorithm')
plt.legend()
plt.show()