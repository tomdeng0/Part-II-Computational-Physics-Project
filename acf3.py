"""Calculates and plots autocorrelation function using statsmodels library"""

import numpy as np
rng = np.random.default_rng()
import matplotlib.pylab as plt
import functions.initial as initial
import functions.metropolis as metropolis
import functions.cluster as cluster

def acf_magnetisation_steps(width, betaJ, wolff=False, MH_n_steps = 500, 
                            MH_burn_sweeps = 200, wolff_n_steps = 200, 
                            wolff_burn_in = 1000):
    N = width**2

    if wolff == True:
        #Wolf ACF
        print('--------------Wolff case--------------')
        p_add = 1 - np.exp(-2*betaJ)
        mag_wolff = []

        print('Equilibrating...')
        lattice = initial.create_lattice(width, 0)
        cluster.n_wolff_moves(lattice, p_add, wolff_burn_in) # equilibration

        print('Generating Wolff series...')
        for step in range(wolff_n_steps):
            cluster.wolff_flip1(lattice, p_add)
            mag_wolff.append(abs(initial.magnetisation(lattice)))
            #if step%100 == 0:
            #    print(step)
        
        print('Calc of Wolff ACF...')
        ACF_wolff = initial.autocorrelation(mag_wolff)
        #ACF_wolff = initial.autocorrelation_2(mag_wolff, len(mag_wolff))

        return ACF_wolff
    
    else:
        # Metropolis-Hastings ACF
        print('--------------Metropolis case--------------')
        mag_MH = []

        print('Equilibrating...')
        lattice = initial.create_lattice(width, 0)
        metropolis.n_MH_moves(lattice, width, betaJ, MH_burn_sweeps*N) # equilibration

        print('Generating MH series...')
        for step in range(MH_n_steps):
            metropolis.MH_flip(lattice, width, betaJ)
            mag_MH.append(abs(initial.magnetisation(lattice)))
            #if step%100 == 0:
            #    print(step)

        print('Calc of Metropolis-Hastings ACF...')
        ACF_MH = initial.autocorrelation(mag_MH)
        #ACF_MH = initial.autocorrelation_2(mag_MH, len(mag_MH))

        return ACF_MH

def acf_magnetisation_sweeps(width, betaJ, n_steps, MH_burn_sweeps = 200, 
                             wolff_burn_steps=2000, wolff=False):
    
    N = width**2

    if wolff == False: 
        
        #------------Metropolis ACF-------------#

        print('--------------Metropolis case--------------')
        
        mag_MH = []
        sweeps_data_MH = []
        MH_burn_in = MH_burn_sweeps*N  # equilibration time for MH

        print('Equilibrating...')
        
        lattice = initial.create_lattice(width, 0)
        metropolis.n_MH_moves(lattice, width, betaJ, MH_burn_in) 

        print('Generating MH series...')
        
        for step in range(n_steps):
    
            sweeps_data_MH.append(step/N)
            metropolis.MH_flip(lattice, width, betaJ)
            mag_MH.append(np.abs(initial.magnetisation(lattice)))

        print('Calculating MH ACF...')
        
        ACF_MH = initial.autocorrelation(mag_MH)
        #ACF_MH = initial.autocorrelation_2(mag_MH, len(mag_MH))

        return ACF_MH, sweeps_data_MH
    
    else: 
        
        #------------Wolff ACF-------------#

        print('--------------Wolff case--------------')
        print('WIDTH = ',width)
        p_add = 1 - np.exp(-2*betaJ)

        #finding average cluster size --- not needed for current setup
        #print('Finding average cluster size...')
        
        #lattice = initial.create_lattice(width, 0)
        #cluster.n_wolff_moves(lattice, p_add, 5000)
        #total_flips = cluster.n_wolff_moves(lattice, p_add, 5000)
        #cluster_size = total_flips/5000

        mag_wolff = []
        sweeps_data_wolff = []

        print('Equilibrating...')
        
        lattice = initial.create_lattice(width, 0)
        cluster.n_wolff_moves(lattice, p_add, wolff_burn_steps)

        print('Generating Wolff series...')
        
        num_flips = 0
        for i in range(n_steps):
            num_flips += cluster.wolff_flip1(lattice, p_add)
            sweeps = num_flips/N
            sweeps_data_wolff.append(sweeps)
            m = np.abs(initial.magnetisation(lattice))
            mag_wolff.append(m)
        
        print('Calculating Wolff ACF...')
        
        ACF_wolff = initial.autocorrelation(mag_wolff)
        #ACF_wolff = initial.autocorrelation_2(mag_wolff, len(mag_wolff))

        return ACF_wolff, sweeps_data_wolff
    
#-------------------------SIMULATION-------------------------------------------

"""
#metropolis
num_avg = 50
MH_acfs = []
MH_sweeps = []
for i in range(num_avg):
    print('STAGE ',i)
    acf_data, sweepdata = acf_magnetisation_sweeps(256,0.441,12800)
    MH_acfs.append(acf_data)
    MH_sweeps.append(sweepdata)
    
print('Averaging...')

MH_acf_sweeps = np.mean(MH_sweeps, axis=0)
MH_acf_averages = np.mean(MH_acfs, axis=0)
MH_acf_stdev = np.std(MH_acfs, axis=0)/np.sqrt(num_avg-1)
MH_sweeps_dev = np.std(MH_sweeps, axis=0)/np.sqrt(num_avg-1)

np.save('MH_acf_256', [MH_acf_sweeps, MH_acf_averages, MH_acf_stdev])

print('Done')
"""

#wolff
num_avg = 20
wolff_acfs = []
#wolff_sweeps = []
for i in range(num_avg):
    print('STAGE ',i)
    #acf_data, sweepdata = acf_magnetisation_sweeps(256,0.441,1000, wolff=True)
    acf_data = acf_magnetisation_steps(512,0.441,True,500,200,700,2000)
    print(len(acf_data))
    wolff_acfs.append(acf_data)
    #wolff_sweeps.append(sweepdata)
    
print('Averaging...')

#wolff_acf_sweeps = np.mean(wolff_sweeps, axis=0)
wolff_acf_averages = np.mean(wolff_acfs, axis=0)
wolff_acf_stdev = np.std(wolff_acfs, axis=0)/np.sqrt(num_avg-1)

#np.save('wolff_acf_256_3', [wolff_acf_sweeps, wolff_acf_averages, wolff_acf_stdev])
np.save('wolff_acf_512_3', [wolff_acf_averages, wolff_acf_stdev])

#metropolis
num_avg = 20
mh_acfs = []
#mh_sweeps = []
for i in range(num_avg):
    print('STAGE ',i)
    #acf_data, sweepdata = acf_magnetisation_sweeps(256,0.441,1000, wolff=True)
    acf_data = acf_magnetisation_steps(256,0.441,False,140000,100,700,2000)
    print(len(acf_data))
    mh_acfs.append(acf_data)
    #mh_sweeps.append(sweepdata)
    
print('Averaging...')

#wolff_acf_sweeps = np.mean(mh_sweeps, axis=0)
mh_acf_averages = np.mean(mh_acfs, axis=0)
mh_acf_stdev = np.std(mh_acfs, axis=0)/np.sqrt(num_avg-1)

#np.save('MH_acf_256_3', [mh_acf_sweeps, mh_acf_averages, mh_acf_stdev])
np.save('MH_acf_256_3', [mh_acf_averages, mh_acf_stdev])