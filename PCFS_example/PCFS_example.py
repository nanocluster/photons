'''
Example code for PCFS data analysis.
'''

import numpy as py
import os
from matplotlib import pyplot as plt
import photons as ph
import PCFS

if __name__ == '__main__':

    path = 'E:/Example/' # path of the folder
    DotD = PCFS.PCFS(path) # create PCFS class
    DotD.get_photons_all() # create all the .photons files


    time_bounds = [1e5, 1e12] # time_window for correlation calculation
    lag_precision = 5
    DotD.get_intensity_correlations(time_bounds, lag_precision) # gets the auto_ and cross_corr, PCFS_interferogram and stores them in the properties

    tau = [5e7, 1e8, 1e9, 1e10] # taus that we're interested in
    xlim = [-3,1] # x window
    white_fringe_ind = 4
    white_fringe = -42.99

    DotD.plot_spectral_diffusion(tau, white_fringe) # plot spectral diffusion at different taus stores them in the properties
    DotD.get_mirror_spectral_corr(white_fringe, white_fringe_ind) # get mirrored spectral correlation stores them in the properties
    DotD.plot_mirror_spectral_corr(tau, xlim) # plot mirrored spectral correlation
