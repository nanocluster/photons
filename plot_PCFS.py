import photons as ph
import PCFS as PCFS
import os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    path = 'C:\\Users\\weiwei\\Downloads\\DotE_run_one'
    dot = PCFS.PCFS(path)
    dot.get_photons_all()
    dot.get_sum_signal_all()

    time_bounds = [1e5, 1e14]
    lag_precision = 5
    if os.path.isfile(path+'\\cross_corr.dat') and os.path.isfile(path+'\\auto_corr.dat') :
        dot.cross_corr_interferogram = np.loadtxt(path+'\\cross_corr.dat')
        dot.auto_corr_sum_interferogram = np.loadtxt(path+'\\auto_corr.dat')
        dot.tau = np.loadtxt(path+'\\tau.dat')
    else:
        dot.get_intensity_correlations(time_bounds, lag_precision)
        np.savetxt(path+'\\cross_corr.dat',dot.cross_corr_interferogram)
        np.savetxt(path+'\\auto_corr.dat',dot.auto_corr_sum_interferogram)
        np.savetxt(path+'\\tau.dat',dot.tau)


    dot.get_blinking_corrected_PCFS()


    tau = [3e7, 1e8, 2e8] #in ps
    wf = -42.305
    # dot.plot_spectral_diffusion(tau, wf)
    # dot.get_mirror_spectral_corr(-42.298,8)
    # dot.plot_mirror_spectral_corr(tau,[-3,3])

    ind0 = np.argmin(abs(dot.tau-tau[0]))
    ind1 = np.argmin(abs(dot.tau-tau[-1]))
    ave_inter = np.mean(dot.blinking_corrected_PCFS_interferogram[ind0:ind1+1,:], axis = 0)
    np.savetxt(path+'\\interferogram.dat',ave_inter)

    path_length_difference = 2*(dot.stage_positions-wf) # in mm
    path_length_time = path_length_difference*10/2.997 # convert to ps
    np.savetxt(path+'\\path_length_time.dat', path_length_time)
    n_path = len(path_length_difference)

    # plot average interferogram
    plt.plot(path_length_time,ave_inter,'-x')
    plt.plot(path_length_time,np.zeros(n_path),'--')
    plt.xlabel('Path length difference [ps]')
    plt.title(dot.PCFS_ID+' PCFS Interferogram averaged over 30-200 us')
    plt.ylabel(r'$g^{(2)}_{cross} - g^{(2)}_{auto}$')
    plt.show()
    plt.close()

    # plot average correlation functions
    ave_cross = np.mean(dot.cross_corr_interferogram[ind0:ind1+1,:], axis = 0)
    ave_auto = np.mean(dot.auto_corr_sum_interferogram[ind0:ind1+1,:], axis = 0)
    plt.plot(path_length_time,ave_cross,'-o', label = 'cross_corr' )
    plt.plot(path_length_time,ave_auto,'-x', label = 'auto_corr' )
    plt.xlabel('Path length difference [ps]')
    plt.title(dot.PCFS_ID+' correlation fuctions averaged over 30-200 us')
    plt.ylabel(r'$g^{(2)}$')
    plt.show()
    plt.close()

    # plot correlation functions
    ind_i = [0,-1]
    for i in range(n_path):
        cross = dot.cross_corr_interferogram[:,i]
        auto = dot.auto_corr_sum_interferogram[:,i]
        plt.semilogx(dot.tau/1e6, cross,'-',label = str(i)+'th cross_corr')
        # plt.semilogx(dot.tau/1e6,auto,'-',label = str(i)+'th auto_corr')
    plt.xlim(1e1,1e6)
    plt.ylim(0.4,1.6)
    plt.xlabel(r'$\tau$ [us]')
    plt.title(dot.PCFS_ID+' cross correlation at different stage position')
    plt.ylabel(r'$g^{(2)}$')
    # plt.legend()
    plt.show()
    plt.close()

    for i in range(n_path):
        cross = dot.cross_corr_interferogram[:,i]
        auto = dot.auto_corr_sum_interferogram[:,i]
        # plt.semilogx(dot.tau/1e6, cross,'-',label = str(i)+'th cross_corr')
        plt.semilogx(dot.tau/1e6,auto,'-',label = str(i)+'th auto_corr')
    plt.xlim(1e1,1e6)
    plt.ylim(0.9,1.2)
    plt.xlabel(r'$\tau$ [us]')
    plt.title(dot.PCFS_ID+' auto correlation at different stage position')
    plt.ylabel(r'$g^{(2)}$')
    # plt.legend()
    plt.show()
    plt.close()
