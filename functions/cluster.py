"""
cluster.py
Defines wolff algorithm and other useful functions for wolff method, 
defines a different evolve_and_plot from initial for choice of both algorithms
"""

import numpy as np
rng = np.random.default_rng()
import matplotlib.pylab as plt
import functions.initial as initial
import functions.metropolis as metropolis
from collections import deque

def wolff_flip1(lattice, p_add):
    """
    Grows the cluster flip by flip with parameter p_add so that it is not 
    computed each step. This avoids having to store the cluster.
    """
    seed = tuple(rng.integers(0,len(lattice),2))
    
    w = len(lattice)
    
    spin = lattice[seed] # get the spin of the seed site
    lattice[seed] = -spin # flip the seed side to begin cluster
    c_size = 1 # initialize tracking of cluster size
    
    remaining_sites = deque([seed]) # track remaining unvisited cluster sites
    while remaining_sites:
        
        site = remaining_sites.pop() # remove from the remaining list
        i, j = site[0], site[1]
        
        for n in initial.get_neighbouring_sites(i,j,w):
            
            if lattice[n] == spin and rng.uniform() < p_add:
                lattice[n] = -spin
                remaining_sites.appendleft(n)
                c_size += 1 # track cluster size
    return c_size

def n_wolff_moves(lattice, p_add, n):
    """
    Carries out `n` wolff moves
    """
    total_flips = 0
    for i in range(n):
        total_flips += wolff_flip1(lattice,p_add)
    return total_flips

def compute_M_avg_wolff(lattice, p_add, avg_times):
    """
    Evolves the lattice using the Wolff algorithm and returns the average 
    absolute magnetisation per site computed using different time steps.
    """
    m=[]
    for t in range(avg_times[-1]+1):
        wolff_flip1(lattice, p_add)
        if t in avg_times:
            m.append(abs(initial.magnetisation(lattice)))
    m_avg = np.mean(m)
    return m_avg

def evolve_and_plot_wolff(lattice, p_add, plot_times):
    """
    Evolves the lattice using MH or Wolff algorithm and plots the lattice
    at different 'time steps'.
    """
    fig, ax = plt.subplots(1, len(plot_times), figsize=(12,4))

    sweeps = 0
    flip_count = 0
    for t in range(plot_times[-1]+1):
        flip_count += wolff_flip1(lattice, p_add)
        if t in plot_times:
            sweeps = flip_count/np.size(lattice)
            initial.plot_lattice(lattice, ax[plot_times.index(t)], "t = {}".format(t))
            ax[plot_times.index(t)].set_xlabel("sweeps = {}".format(sweeps))
    plt.show()