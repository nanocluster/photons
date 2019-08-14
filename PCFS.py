'''
Python for analysis of photon resolved PCFS file as generated from the Labview instrument control softeware by Hendrik Utzat V3.0.

Adapted from PCFS.m V4.0 @ Hendrik Utzat 2017.

Weiwei Sun, July, 2019
'''

import numpy as np
import time as timing
import os, struct, scipy, re, glob
import photons as ph
from matplotlib import pyplot as plt


class PCFS:

    '''
    Called when creating a PCFS object. This function saves the fundamental arguments as object properties.
    folder_path is the full path to the folder containin the photon stream data and meta data of the PCFS run. NOTE: '\' needs to be replaced to '\\' or '/' and the not ending with '/', i.e., 'D/Downloads/PCFS/DotA'.
    memory_limit is the maximum memory to read metadata at once, set to default 1 MB.
    '''
    def __init__(self, folder_path, memory_limit = 1, header_lines_pcfslog = 5):

        tic = timing.time()
        # property
        self.cross_corr_interferogram = None
        self.auto_corr_sum_interferogram = None
        self.tau = None
        self.PCFS_interferogram = None
        self.blinking_corrected_PCFS_interferogram = None
        self.spectral_correlation = {}
        self.Fourier = {}
        self.memory_limit = memory_limit

        # extract files information
        self.path_str = folder_path
        self.PCFS_ID = os.path.split(folder_path)[1] # extract the folder name, i.e., 'DotA'
        os.chdir(folder_path)
        file_pos = glob.glob('*.pos')
        if len(file_pos) == 0:
            print('.pos file not found!')
        self.file_pos = file_pos[0] # with extension

        file_pcfslog = glob.glob('*.pcfslog')
        if len(file_pcfslog) == 0:
            print('.pcfslog file not found!')
        self.file_pcfslog = file_pcfslog[0] # with extension

        self.file_stream = [f.rstrip('.stream') for f in glob.glob('*.stream')] # without extension



        # read in the position file as array
        self.get_file_photons() # get all the photons files in the current directory
        self.stage_positions = np.loadtxt(self.file_pos)
        # read in the metadata of the .pcfslog file and store it as property
        with open(self.file_pcfslog) as f:
            lines_skip_header = f.readlines()[header_lines_pcfslog:]
        self.pcfslog = {}
        for lines in lines_skip_header:
            lines_split = lines.split('=')
            if len(lines_split) == 2:
                self.pcfslog[lines_split[0]] = float(lines_split[1])

        #create photons object for the photon stream at each interferometer path length difference.
        self.photons = {}
        for f in self.file_stream:
            self.photons[f] = ph.photons(f+'.stream', self.memory_limit)

        toc= timing.time()
        print('Total time elapsed is %4f s' % (toc - tic))


    '''
    ============================================================================================
    Get and parse photon stream / get correlation functions
    '''


    # This function gets all the photon files in the current directory
    def get_file_photons(self):
        self.file_photons = [f.rstrip('.photons') for f in glob.glob('*.photons')] # without extension



    '''
    This function gets all photon stream data.
    '''
    def get_photons_all(self):

        time_start = timing.time()
        self.get_file_photons()
        for f in self.file_stream:
            if f not in self.file_photons:
                print(f)
                self.photons[f].get_photon_records(memory_limit = self.memory_limit)
        time_end= timing.time()
        print('Total time elapsed is %4f s' % (time_end - time_start))



    '''
    This function gets the sum signal of the two detectors for all photon arrival files.
    '''
    def get_sum_signal_all(self):
        time_start = timing.time()
        self.get_file_photons()
        for f in self.file_photons:
            if 'sum' not in f:
                self.photons[f].write_photons_to_one_channel(f, 'sum_signal_'+f)
        time_end= timing.time()
        print('Total time elapsed is %4f s' % (time_end - time_start))



    '''
    This function obtains the cross-correlations and the auto-correlation of the sum signal at each stage position and returns them in a matrix self.cross_corr_interferogram and the auto-correlation function of the sum signal to a matirx of similar structure.
    '''
    def get_intensity_correlations(self, time_bounds, lag_precision):
        time_start= timing.time()
        self.get_file_photons()
        self.time_bounds = time_bounds
        self.lag_precision = lag_precision
        cross_corr_interferogram = None
        auto_corr_sum_interferogram = None

        if len(self.file_photons) == len(self.file_stream):
            self.get_sum_signal_all()
        self.get_file_photons()

        for f in self.file_photons:
            # looking to get cross correlation
            print(f)
            if 'sum' not in f:
                if self.photons[f].cross_corr is None:
                    self.photons[f].photon_corr(f, 'cross', [0,1], time_bounds, lag_precision, 0)
                if self.tau is None:
                    self.tau = self.photons[f].cross_corr['lags']
                    self.length_tau = len(self.tau)

                # create an array containing to be filled up with the PCFS interferogram.
                if cross_corr_interferogram is None:
                    cross_corr_interferogram = np.zeros((self.length_tau, len(self.stage_positions)))

                correlation_number = int(re.findall(r'\d+', f)[-1]) # extract the number of correlation measurements from the file names
                cross_corr_interferogram[:, correlation_number] = self.photons[f].cross_corr['corr_norm']

            # looking to get auto-correlation for sum signals
            else:
                if self.photons[f[11:]].auto_corr is None:
                    self.photons[f[11:]].photon_corr(f, 'auto', [0,0], time_bounds, lag_precision, 0)
                if self.tau is None:
                    self.tau = self.photons[f[11:]].auto_corr['lags']
                    self.length_tau = len(self.tau)
                # create an array containing to be filled up with the PCFS interferogram.
                if auto_corr_sum_interferogram is None:
                    auto_corr_sum_interferogram = np.zeros((self.length_tau, len(self.stage_positions)))

                correlation_number = int(re.findall(r'\d+', f)[-1]) # extract the number of correlation measurements from the file names
                auto_corr_sum_interferogram[:, correlation_number] = self.photons[f[11:]].auto_corr['corr_norm']
            print('==============================')

        self.cross_corr_interferogram = cross_corr_interferogram.copy()
        self.auto_corr_sum_interferogram = auto_corr_sum_interferogram.copy()

        # substract auto-correlation of sum signal from the cross correlation.
        PCFS_interferogram = cross_corr_interferogram - auto_corr_sum_interferogram
        self.PCFS_interferogram = PCFS_interferogram.copy()


        time_end= timing.time()
        print('Total time elapsed is %4f s' % (time_end - time_start))



    '''
    ============================================================================================
    Analysis of data
    '''

    '''
    This function gets blinking corrected PCFS interferogram.
    '''
    def get_blinking_corrected_PCFS(self):
        self.blinking_corrected_PCFS_interferogram = 1 - self.cross_corr_interferogram / self.auto_corr_sum_interferogram


    '''
    This function gets and plots spectral diffusion
    '''
    def plot_spectral_diffusion(self, tau_select, white_fringe):
        if self.get_blinking_corrected_PCFS is None:
            self.get_blinking_corrected_PCFS()

        # plot PCFS interferogram at different tau
        x = 2 * (self.stage_positions - white_fringe) # in mm
        ind = np.array([np.argmin(np.abs(self.tau - tau)) for tau in tau_select])
        legends = [tau/1e9 for tau in tau_select]
        y = self.blinking_corrected_PCFS_interferogram[ind, :]
        x = x/(3e8)/1000*1e12 # convert to ps

        plt.figure()
        plt.subplot(3,1,1)
        for i in range(len(tau_select)):
            plt.plot(x, y[i,:])

        plt.ylabel(r'$g^{(2)}_{cross} - g^2_{auto}$')
        # plt.xlabel('Optical Path Length Difference [mm]')
        plt.xlabel('Optical Path Length Difference [ps]')
        plt.legend(legends)
        plt.title(self.PCFS_ID + r' PCFS Interferogram at $\tau$ [ms]')

        plt.subplot(3,1,2)
        for i in range(len(tau_select)):
            plt.plot(x, y[i,:]/max(y[i,:]))

        plt.ylabel(r'$g^{(2)}_{cross} - g^2_{auto}$')
        # plt.xlabel('Optical Path Length Difference [mm]')
        plt.xlabel('Optical Path Length Difference [ps]')
        plt.legend(legends)
        plt.title('Normalized ' + self.PCFS_ID + r' PCFS Interferogram at $\tau$ [ms]')

        plt.subplot(3,1,3)
        for i in range(len(tau_select)):
            plt.plot(x, np.sqrt(y[i,:]/max(y[i,:])))

        plt.ylabel(r'$g^{(2)}_{cross} - g^2_{auto}$')
        # plt.xlabel('Optical Path Length Difference [mm]')
        plt.xlabel('Optical Path Length Difference [ps]')
        plt.legend(legends)

        plt.title('Squared root of Normalized ' + self.PCFS_ID + r' PCFS Interferogram at $\tau$ [ms]')
        plt.tight_layout()
        plt.show()

        
        
    '''
    ==========================================================================================
    Depreciated
    '''

    
    '''
    This function gets spectral correlation data. Depreciated.
    '''
    def plot_mirror_spectral_corr(self, tau_select, xlim):
        x = self.mirror_spectral_correlation['zeta']
        ind = np.array([np.argmin(np.abs(self.tau - tau)) for tau in tau_select])
        legends = [tau/1e9 for tau in tau_select]
        y = self.mirror_spectral_correlation['spectral_corr'][ind,:]

        plt.figure()
        plt.subplot(2,1,1)
        for i in range(len(tau_select)):
            plt.plot(x, y[i,:])

        plt.ylabel(r'$p(\zeta)$')
        plt.xlabel(r'$\zeta$ [meV]')
        plt.xlim(xlim)
        plt.legend(legends)
        plt.title(self.PCFS_ID + r' Mirrored Spectral Correlation at $\tau$ [ms]')

        plt.subplot(2,1,2)
        for i in range(len(tau_select)):
            plt.plot(x, y[i,:]/max(y[i,:]))

        plt.ylabel(r'Normalized $p(\zeta)$')
        plt.xlabel(r'$\zeta$ [meV]')
        plt.xlim(xlim)
        plt.legend(legends)

        plt.title(self.PCFS_ID + r' Mirrored Spectral Correlation at $\tau$ [ms]')
        plt.tight_layout()
        plt.show()



    '''
    This function gets mirrored spectral correlation by interpolation. Depreciated.
    '''
    def get_mirror_spectral_corr(self, white_fringe_pos, white_fringe_ind):

        # construct mirrored data
        interferogram = self.blinking_corrected_PCFS_interferogram[:,:]
        mirror_intf = np.hstack((np.fliplr(interferogram[:, white_fringe_ind:]), interferogram[:, white_fringe_ind+1:]))
        temp = white_fringe_pos - self.stage_positions[white_fringe_ind:]
        temp = temp[::-1]
        mirror_stage_pos = np.hstack((temp, self.stage_positions[white_fringe_ind+1:] - white_fringe_pos))
        interp_stage_pos = np.arange(min(mirror_stage_pos), max(mirror_stage_pos)+0.01, 0.01 )

        # row-wise interpolation
        a,b = mirror_intf.shape
        interp_mirror = np.zeros((a,len(interp_stage_pos)))
        for i in range(a):
            interp_mirror[i,:] = np.interp(interp_stage_pos, mirror_stage_pos, mirror_intf[i,:])

        self.mirror_stage_pos = mirror_stage_pos
        self.mirror_PCFS_interferogram = interp_mirror # not including the first line of position

        #some constants
        eV2cm = 8065.54429
        cm2eV = 1 / eV2cm

        N = len(interp_stage_pos)
        path_length_difference = 0.2 * (interp_stage_pos) # NOTE: This is where we convert to path length difference space in cm.
        delta = (max(path_length_difference) - min(path_length_difference)) / (N-1)
        zeta_eV = np.fft.fftshift(np.fft.fftfreq(N, delta)) * cm2eV * 1000 # in meV

        # get reciprocal space (wavenumbers).
        # increment = 1 / delta
        # zeta_eV = np.linspace(-0.5 * increment, 0.5 * increment, num = N) * cm2eV * 1000 # converted to meV

        # take the FFT of the interferogram to get the spectral correlation. All that shifting is to shift the zero frequency component to the middle of the FFT vector. We take the real part of the FFT because the interferogram is by definition entirely symmetric.
        spectral_correlation = self.mirror_PCFS_interferogram.copy()
        for i in range(a):
            spectral_correlation[i,:] = np.abs(np.fft.fftshift(np.fft.fft(self.mirror_PCFS_interferogram[i,:])))

        self.mirror_spectral_correlation = {}
        self.mirror_spectral_correlation['spectral_corr'] = spectral_correlation
        self.mirror_spectral_correlation['zeta'] = zeta_eV




    '''
    This function calculates and plots the spectral correlation of an interterferogram parsed as two vectors containing the stage_positions (not path length differences!), the corresponding interferogram values and the white-fringe position.
    '''
    def plot_spectral_corr(self, stage_positions, interferogram, white_fringe_pos):

        #some constants
        eV2cm = 8065.54429
        cm2eV = 1 / eV2cm

        N = len(stage_positions)
        path_length_difference = 2 * (stage_positions - white_fringe_pos) * 0.1 # NOTE: This is where we convert to path length difference space in cm.
        delta = (max(path_length_difference) - min(path_length_difference)) / (N-1)
        zeta_eV = np.fft.fftshift(np.fft.fftfreq(N, delta)) * cm2eV * 1000 # in meV

        # get reciprocal space (wavenumbers).
        # increment = 1 / delta
        # zeta_eV = np.linspace(-0.5 * increment, 0.5 * increment, num = N) * cm2eV * 1000 # converted to meV

        # take the FFT of the interferogram to get the spectral correlation. All that shifting is to shift the zero frequency component to the middle of the FFT vector. We take the real part of the FFT because the interferogram is by definition entirely symmetric.
        spectral_correlation = np.abs(np.fft.fftshift(np.fft.fft(interferogram)))

        normalized_spectral_correlatoin = spectral_correlation / max(spectral_correlation)


        plt.plot(zeta_eV, normalized_spectral_correlatoin, '-o', markersize = 1)

        plt.ylabel(r'Normalized $p(\zeta)$')
        plt.xlabel(r'$\zeta$ [meV]')

        plt.title(self.PCFS_ID + r' Spectral Correlation at $\tau$ [ms]')
        plt.show()



    '''
    This function gets the fourier spectrum from the photon stream.
    '''
    def get_Fourier_spectrum_from_stream(self, bin_width, file_in):
        t = np.zeros(len(self.stage_positions)) # for intensity
        Fourier = np.zeros(len(self.stage_positions))
        self.get_file_photons()

        for f in self.file_photons:
            # looking to get cross correlation
            if 'sum' not in f:
                if file_in in f:
                    correlation_number = int(re.findall(r'\d+', f)[0]) # extract the number of correlation measurements from the file names
                    self.photons[f].get_intensity_trace(f, bin_width)
                    intensity = self.photons[f].intensity_counts['Trace']
                    t[correlation_number] = (np.sum(intensity[:,0]) + np.sum(intensity[:,1]))
                    Fourier[correlation_number] = (np.sum(intensity[:,0]) - np.sum(intensity[:,1])) / t[correlation_number]

        out_dic = {}
        out_dic['Fourier'] = Fourier
        out_dic['stage_pos'] = self.stage_positions
        out_dic['intensity'] = t
        self.Fourier[file_in] = out_dic
