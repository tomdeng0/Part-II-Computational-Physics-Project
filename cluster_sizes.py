"""generates data for cluster size as a function of temperature"""

import numpy as np
rng = np.random.default_rng()
import functions.initial as initial
import functions.cluster as cluster

def av_cluster_sizes(lattice, p_add, burn_in, n_moves):
    
    cluster.n_wolff_moves(lattice, p_add, burn_in)
    
    flips = []
    cluster_size = 0
    for i in range(n_moves):
        num_flips = cluster.n_wolff_moves(lattice,p_add,1)
        flips.append(num_flips)
    cluster_size = np.mean(flips)
    error = np.std(flips)/np.sqrt(n_moves-1)

    return cluster_size, error

width = 40
N = width**2
temps = np.linspace(1.5,3.5,30)
burn_in = 500
n_moves = 500

cluster_sizes = []
errors = []

for temp in temps:
    print(temp)
    p_add = 1 - np.exp(-2/temp)
    lattice = initial.create_lattice(width, 0)
    cluster_size, error = av_cluster_sizes(lattice, p_add, burn_in, n_moves)
    cluster_sizes.append(cluster_size)
    errors.append(error)

data = [temps,cluster_sizes,error]
np.save('cluster_sizes_data_2', data)