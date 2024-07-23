import numpy as np
rng = np.random.default_rng()
import matplotlib.pylab as plt
import functions.initial as initial
import functions.metropolis as metropolis
import functions.cluster as cluster

width = 40  # width of the lattice
N = width**2
n_sweeps = 200
betaJ = 0.441

#------------Metropolis ACF-------------#

print('--------------Metropolis case--------------')
mag_MH = []
sweeps_data_MH = []
MH_n_steps = n_sweeps*N  # number of steps to run each simulation for Metropolis-Hastings algorithm
MH_burn_in = 500*N  # equilibration time for Metropolis-Hastings algorithm

print('Equilibrating...')
lattice = initial.create_lattice(width, 0)
metropolis.n_MH_moves(lattice, width, betaJ, MH_burn_in) 

print('Generating MH series...')
for step in range(MH_n_steps):
    
    sweeps_data_MH.append(step/N)
    metropolis.MH_flip(lattice, width, betaJ)
    mag_MH.append(np.abs(initial.magnetisation(lattice)))

print('Calculating MH ACF...')
ACF_MH = initial.autocorrelation(mag_MH)
#ACF_MH_2 = initial.autocorrelation_2(mag_MH, len(mag_MH))

#------------Wolff ACF-------------#

print('--------------Wolff case--------------')
p_add = 1 - np.exp(-2*betaJ)

#finding average cluster size
print('Finding average cluster size...')
lattice = initial.create_lattice(width, 0)
cluster.n_wolff_moves(lattice, p_add, 1000)
total_flips = cluster.n_wolff_moves(lattice, p_add, 1000)
cluster_size = total_flips/1000

mag_wolff = []
sweeps_data_wolff = []
wolff_n_steps = 200 # number of moves for wolff algorithm
wolff_burn_in = 2000 # equilibration time for wolff

print('Equilibrating...')
lattice = initial.create_lattice(width, 0)
cluster.n_wolff_moves(lattice, p_add, wolff_burn_in)

print('Generating Wolff series...')
num_flips = 0
sweeps = 0
for i in range(wolff_n_steps):
    num_flips += cluster.wolff_flip1(lattice, p_add)
    if num_flips > N:
        sweeps += num_flips/N
        #print(sweeps)
        sweeps_data_wolff.append(sweeps)
        m = np.abs(initial.magnetisation(lattice))
        mag_wolff.append(m)
        num_flips = 0

print('Calculating Wolff ACF...')
ACF_wolff = initial.autocorrelation(mag_wolff)
#ACF_wolff_2 = initial.autocorrelation_2(mag_wolff, len(mag_wolff))

#data=[ACF_MH,ACF_wolff]
#np.save('acf_accountforsweeps', data)

print('wolff acf length = ', len(ACF_wolff))
print('wolff series length = ', len(sweeps_data_wolff))
print('mh acf length = ', len(ACF_MH))
print('mh series length = ', len(sweeps_data_MH))

plt.figure()
plt.plot(sweeps_data_wolff, ACF_wolff, label='Wolff', color='blue')
#plt.plot(sweeps_data_wolff, ACF_wolff_2, label='Wolff2', color='navy')
plt.plot(sweeps_data_MH, ACF_MH, label='Metropolis-Hastings', color='red')
#plt.plot(sweeps_data_MH, ACF_MH_2, label='Metropolis-Hastings2', color='firebrick')
#plt.yscale("log")
plt.axhline(y=np.exp(-1), color='black', linestyle='--', label='1/e')
plt.xlabel('Sweeps of the latttice')
plt.ylabel('ACF')
plt.title(r'ACF for MH vs Wolff at critical temperature')
plt.legend()
plt.show()