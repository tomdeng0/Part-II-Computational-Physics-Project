"""Plots the analyitical and wolff simulated solutions for the average magnetisation of the lattice
"""
import numpy as np
rng = np.random.default_rng()
import matplotlib.pylab as plt
import functions.initial as initial
import functions.cluster as cluster

betaJ_analytical = np.linspace(0,1.5,200)
beta_crit = 0.5*np.log(1+np.sqrt(2))

#defining analytical absolute magnetisation solution
def Mag(x):
    if x <= beta_crit:
        return 0
    else:
        return ((1 - np.sinh(2*x)**(-4))**(1/8))

x = betaJ_analytical  
M_analytical = np.zeros(len(x))

#generating abs analytical
for i in range(len(x)):
    M_analytical[i] = Mag(x[i])

# different betaJ
width = 25
N = width**2
betaJ_mcmc = np.linspace(0,1.1,50)
M_mcmc = []
M_error =[]
burn_in = 200
n_moves = 5000

for i in betaJ_mcmc:
    lattice = initial.create_lattice(width, type=0)
    p_add = 1-np.exp(-2*i)
    print('beta = ' + str(i))
    cluster.n_wolff_moves(lattice, p_add, burn_in)
    mag = []
    for j in range(n_moves):
        cluster.wolff_flip1(lattice, p_add)
        mag.append(abs(initial.magnetisation(lattice)))
        if j%100 == 0:
            print(j)
    magnetisation, error = initial.batch_estimate(mag, lambda x: np.mean(x), num_batches=-1, batch_with_autocorr=True)
    M_mcmc.append(magnetisation)
    M_error.append(error)

data_analytical = [betaJ_analytical, M_analytical]
data_mcmc = [betaJ_mcmc, M_mcmc, M_error]

np.save('wolff_onsager_data', data_mcmc)
np.save('analytical_onsager_data', data_analytical)

fig = plt.figure()
plt.plot(betaJ_analytical, M_analytical, label='Analytical', color='b')
plt.plot(betaJ_mcmc, M_mcmc, label='Wolff MC Method', color='red')
plt.xlabel (r'Inverse temperature $\beta$')
plt.ylabel ('Absolute Magnetisation, |M|')
plt.legend(loc='upper left')
plt.title(r'Simulated (Wolff) and anaytical solutions for $\langle |M| \rangle$ against $\beta$')
#plt.savefig('wolff_onsager.png')
plt.show()