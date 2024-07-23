# Part II Computational Physics Project

see workbook.ipynb for the report

## Abstract

This report explores the efficiency and performance of cluster algorithms for Markov Chain Monte Carlo (MCMC) methods specifically for the case of the 2-dimensional Ising model. In particular, the Wolff algorithm is implemented, analysed and compared to the Metropolis-Hastings (MH) algorithm. The performance of these algorithms is studied near criticality, exploring autocorrelation times and the phenomenon of critical slowing down via scaling the size of the lattice. The implementation of the algorithms was successful and the plots generated consistenly showed the Wolff algorithm to outperform the MH algorithm in dealing with slowing down and large cluster sizes at the phase transition. The Wolff algorithm produced simulated solutions close to the exact results compared with the MH case which struggled around the transition point.
