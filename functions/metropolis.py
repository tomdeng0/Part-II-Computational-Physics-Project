"""
metropolis.py
Defines Metropolis-Hastings algorithm
"""

import functions.initial as initial
import numpy as np
rng = np.random.default_rng()
import matplotlib.pylab as plt

def compute_acceptance(i, j, lattice, width, betaJ):
    """Computes the acceptance probability from energy difference between the old and 
       new state if spin [i,j] would be flipped.
    """
    betaDeltaE=2*(betaJ*lattice[i,j]*initial.neighbouring_spins_sum(i, j, lattice, width))
    acceptance=np.exp(-1*betaDeltaE)
    return acceptance, betaDeltaE

def MH_flip(lattice, width, betaJ):
    """Proposes a new site to be flipped (proposal matrix) and accepts or rejects the flip based on MH acceptance matrix"""
    i, j = np.random.randint(0,width,2)

    acceptance, betaDeltaE=compute_acceptance(i, j, lattice, len(lattice), betaJ)

    #comparing probabilities
    if betaDeltaE <= 0:
        lattice[i,j]= -1*lattice[i,j]
    else:
        if acceptance > np.random.rand():
            lattice[i,j]*= -1

def n_MH_moves(lattice, width, betaJ, n):
    for i in range(n):
        MH_flip(lattice, width, betaJ)

def evolve_and_plot_MH(lattice, width, betaJ, plot_times):
    """Evolves the lattice using MH algorithm and plots the lattice at different 'time steps'."""
    fig, ax = plt.subplots(1, len(plot_times), figsize=(16,6))
    
    for t in range(plot_times[-1]+1):
        MH_flip(lattice, width, betaJ)
        if t in plot_times:
            initial.plot_lattice(lattice, ax[plot_times.index(t)], "t = {}".format(t))
            ax[plot_times.index(t)].set_xlabel("sweeps = {}".format(t/width**2))
    plt.show()