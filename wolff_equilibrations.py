"""Plots magnetisation against number of wolff moves at different temps to 
   illustrate equilibration of the algorithm
"""

import numpy as np
rng = np.random.default_rng()
import functions.initial as initial
import functions.cluster as cluster

def equilibration_wolff(width, betaJ, n_steps, filename):   
    
    p_add = 1-np.exp(-2*betaJ)
    lattice_types = [(0, "Random uniform"), (1, "All up"), (-1, "All down"), (2, "Anti-aligned")]
    magnetisation_data = []
    for i, (lattice_type, name) in enumerate(lattice_types):
        lattice = initial.create_lattice(width, type=lattice_type)
        print(i)

        # Measurement stage
        N = width**2
        mag = np.abs(initial.magnetisation(lattice))
        magnetisations = [mag]
        for j in range(n_steps):
            cluster.wolff_flip1(lattice, p_add)
            if j in np.arange(n_steps)*5:
                mag = np.abs(initial.magnetisation(lattice))
                magnetisations.append(mag)

        magnetisation_data.append([name, magnetisations])

    np.save(filename, np.array(magnetisation_data, dtype=object))

equilibration_wolff(40,0.44,1000, 'wolff_equilibration_data_0.44')
equilibration_wolff(40,0.33,1000, 'wolff_equilibration_data_0.33')
equilibration_wolff(40,1.00,1000, 'wolff_equilibration_data_1.00')