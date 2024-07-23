import numpy as np
rng = np.random.default_rng()
import matplotlib.pylab as plt
import functions.initial as initial
import functions.metropolis as metropolis
import functions.cluster as cluster

def evolve_and_plot_MH(lattice, width, betaJ, plot_times):
    """Evolves the lattice using MH algorithm and plots the lattice at different 'time steps'."""
    fig, ax = plt.subplots(1, len(plot_times), figsize=(16,6))
    lattices_data = np.zeros((len(plot_times),width,width))
    sweeps_data = []
    
    for t in range(plot_times[-1]+1):
        metropolis.MH_flip(lattice, width, betaJ)
        if t in plot_times:
            print(t)
            lat_freeze = lattice
            lattices_data[plot_times.index(t)] = lat_freeze
            sweeps = t/width**2
            sweeps_data.append(sweeps)
            initial.plot_lattice(lattice, ax[plot_times.index(t)], "t = {}".format(t))
            ax[plot_times.index(t)].set_xlabel("sweeps = {}".format(sweeps))
    plt.show()
    data = [lattices_data.tolist(), plot_times, sweeps_data]
    return data

def evolve_and_plot_wolff(lattice, p_add, plot_times):
    """Evolves the lattice using MH or Wolff algorithm and plots the lattice at different 'time steps'."""
    fig, ax = plt.subplots(1, len(plot_times), figsize=(16,6))
    lattices_data = np.zeros((len(plot_times),width,width))
    sweeps_data = []

    sweeps = 0
    flip_count = 0
    for t in range(plot_times[-1]+1):
        flip_count += cluster.wolff_flip1(lattice, p_add)
        if t in plot_times:
            print(t)
            lat_freeze = lattice
            lattices_data[plot_times.index(t)] = lat_freeze
            sweeps = flip_count/np.size(lattice)
            sweeps_data.append(sweeps)
            initial.plot_lattice(lattice, ax[plot_times.index(t)], "t = {}".format(t))
            ax[plot_times.index(t)].set_xlabel("sweeps = {}".format(sweeps))
    plt.show()
    data = [lattices_data.tolist(), plot_times, sweeps_data]
    return data

width = 100
betaJ = 1
p_add = 1 - np.exp(-2*betaJ)
initial_lattice = initial.create_lattice(width,0)

#plot_times=[0,100,1000,10000,100000,200000,500000]
#MH_lattices_data  = evolve_and_plot_MH(initial_lattice, width, betaJ, plot_times)
#print(MH_lattices_data)
#np.save('MH_states_100_0.20', MH_lattices_data)

plot_times = [0,100,1000,10000,20000,50000,75000]
wolff_lattices_data = evolve_and_plot_wolff(initial_lattice, p_add, plot_times)
np.save('wolff_states_100_1.00', wolff_lattices_data)