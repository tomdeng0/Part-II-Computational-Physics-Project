"""
initial.py
Creates a lattice and defines useful functions that calculate 
neighbouring sites, the sum of neighbouring spin sites, the total 
magnetisation of the lattice and plot the lattice. The autocorrelation 
function and the finding the autocorrelation times have also been 
defined with and without the 'statsmodels' library
"""

import numpy as np
rng = np.random.default_rng()
import matplotlib.pylab as plt
from statsmodels.tsa.stattools import acf
import types
import math

def create_lattice(width,type=0):
    """
    Creates a lattice array of four types
    
    Type 1: all spins +1
    Type -1: all spins -1
    Type 0: a random lattice of spins +/- 1
    Type 2: anti-aligned lattice
    """
    lattice = np.array((width,width))   
    if type == 1:
        return np.ones((width,width))
    elif type == 0:
        return np.random.choice([1,-1],(width,width))
    elif type == -1:
        return (-1)*np.ones((width,width))
    elif type == 2:
            if width%2 == 0:
                return np.tile([[1,-1],[-1,1]],(width//2,width//2))
            else:
                return np.tile([[1,-1],[-1,1]],((width+1)//2,(width+1)//2))[:width,:width]
    else:
        raise ValueError('Invalid type. Type should be 0, 1, or -1.')

def plot_lattice(lattice,ax,title):
    """Plots the lattice with black as +1, and white as -1"""
    ax.matshow(lattice, vmin=-1, vmax=1, cmap=plt.cm.binary)
    ax.title.set_text(title)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_xticks([])

def get_neighbouring_sites(i,j,width):
    """
    Returns the coordinates of the neighbouring sites of a given 
    coordinate with periodic boundrary conditions
    """
    neighbours = [((i-1)%width, j), ((i+1)%width, j), (i, (j-1)%width), (i, (j+1)%width)]
    return neighbours

def neighbouring_spins_sum(i,j,lattice,width):
    """Returns the sum of the spins of the neighbouring sites of a given coordinate (i,j)"""
    neighbours = get_neighbouring_sites(i,j,width)
    sum_neighbours = sum(lattice[i] for i in neighbours)
    return sum_neighbours

def magnetisation(lattice):
    """Computes the overall magnetisation of the lattice"""
    Mag = lattice.sum()/np.size(lattice)
    return Mag

"""
Defining autocorrelation function and autocorrelation times 
using 'statsmodels' library
"""

def autocorrelation(data):
    """
    Returns the autocorrelation function of a data series, if a value
    is found as nan when calculating acf, returns 0 for that entry
    """
    N = len(data)
    autocorr = acf(data,adjusted=True,fft=False, nlags=(N-1))
    for i in range(len(autocorr)):
        if math.isnan(autocorr[i]) == True:
            autocorr[i] = 0
    return autocorr

def autocorrelation_time(data, data_type_is_autocorr=bool):
    """
    Returns the autocorrelation time for either a given data series
    or for a given autocorrelation function

    `data_type_is_autocorr = True` for an acf input
    `data_type_is_autocorr = False` for another data series input
    """
    if data_type_is_autocorr == False:
        autocorr = autocorrelation(data)
        crit = np.exp(-1)
        t_a = np.argmin(autocorr>crit, axis=0)
        return t_a if t_a>0 else len(autocorr)
        
    else:
        crit = np.exp(-1)
        t_a = np.argmin(data<crit, axis=0)
        return t_a if t_a>0 else len(data)


def batch_estimate(data, operation, num_batches, batch_with_autocorr):
    """
    Calculates a batch estimate of a data series using the autocorrelation time
    
    `operation` parameter must be a function
    
    `batch_with_autocorr = True` uses the autocorrelation time to determine `num_batches`
    
    `batch_with_autocorr = False` uses `num_batches` whuch must be specified
    """
    if batch_with_autocorr == True:
        t_a = autocorrelation_time(data)
        print(t_a)
        if t_a == 0:
            m = int(len(data)/2)
        else: 
            m = int(len(data)/(2*t_a))

    elif batch_with_autocorr == False:
        m = num_batches

    else:
        raise ValueError("Invalid entry. batch_with_autocorr should be True or False")
    
    if isinstance(operation, types.FunctionType) == False:
        raise ValueError("Invalid entry, operation variable must be a function on an array")
    
    batches = np.array_split(data, m)
    assign = np.array(list(map(operation, batches)))
    estimate = np.mean(assign)
    error = np.std(assign)/np.sqrt(m-1)
    return estimate, error

#############################################################################

"""
Defining autocorrelation function and autocorrelation times
without 'statsmodels' library
"""

def autocorrelation_2(data,t_max):
    """Computes ACF for a given time series"""
    size = len(data)
    mean = np.mean(data)
    t_max = min((t_max,size))
    # autocovariance
    autocov = np.zeros(size)
    for t in range(t_max):
        autocov[t] = np.dot(data[:size-t] - mean, data[t:] - mean) / (size-t)

    #normalise
    autocorr = autocov/autocov[0]

    return autocorr

def autocorrelation_time_2(data, tmax):
    """Finds the autocorrelation time of a given data series"""
    autocorr = autocorrelation_2(data, tmax)
    crit = np.exp(-1)
    time = np.argmin(autocorr>crit, axis=0)
    return time