"""Calculate and plot dynamic exponent via finite size scaling"""

import numpy as np
import matplotlib.pylab as plt
import functions.initial as initial
from scipy.stats import linregress
import acf3 as acf_gen

def compute_dynamic_exponent_MH(w_values, ac_times, filename, generate_needed):
    
    if generate_needed == True:
        print('Generating data for widths = ', w_values)
        beta = 0.4407
        ACF_MH_averages = []
        #metropolis
        for w in w_values:
            print('width = ', w)
            acf1 = acf_gen.acf_magnetisation_steps(w,beta,False)
            acf2 = acf_gen.acf_magnetisation_steps(w,beta,False)
            acf3 = acf_gen.acf_magnetisation_steps(w,beta,False)
            acf4 = acf_gen.acf_magnetisation_steps(w,beta,False)
            acf5 = acf_gen.acf_magnetisation_steps(w,beta,False)

            print('Averaging ACFs...')
            ACF_MH_averages.append(np.mean(([acf1,acf2,acf3,acf4,acf5]), axis=0))

        np.save(filename, [ACF_MH_averages, w_values])

        act_values = []
        for i in range(len(ACF_MH_averages)):
            t_a = initial.autocorrelation_time(ACF_MH_averages[i], True)
            act_values.append(t_a)
    else:
        act_values = ac_times
    
    log_w = np.log(w_values)
    log_tau = np.log(act_values)
    coeffs = linregress(log_w,log_tau)
    z = coeffs.slope
    c = coeffs.intercept
    
    plt.figure()
    plt.plot(log_w, log_tau, label='data')
    plt.plot(log_w, c + z*log_w, label='fitted line, z = {0:.2f} $\pm$ {1:.2f}'.format(z, coeffs.stderr))
    plt.legend()
    plt.show()

    return z, coeffs.stderr

"""
beta = 0.44
widths = [16,32,64,128,256,512]
ACF_MH_averages = []
#metropolis
for w in widths:
    print('width = ', w)
    acf1 = acf_series(w,beta,False)
    acf2 = acf_series(w,beta,False)
    acf3 = acf_series(w,beta,False)
    acf4 = acf_series(w,beta,False)
    acf5 = acf_series(w,beta,False)

    print('Averaging ACFs...')
    ACF_MH_averages.append(np.mean(([acf1,acf2,acf3,acf4,acf5]), axis=0))

np.save('acf_data_widths',[ACF_MH_averages,widths])

ac_times = []
for i in range(len(ACF_MH_averages)):
    t_a = initial.autocorrelation_time(ACF_MH_averages[i], True)
    ac_times.append(t_a)

for i in range(len(ACF_MH_averages)):
    plt.plot(ACF_MH_averages[i], label=widths[i])
plt.xlim(0,2*np.mean(ac_times))
plt.legend()
plt.yscale("log")
plt.show()
"""