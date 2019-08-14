'''
This function fits the spectral correlation to multiple lorentzians.
'''

import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

'''
These functions are in the spectral domain.
'''

def fit_with_lorentzian(zeta_in, spectral_corr, params):
    if len(params) == 4:
        popt, pconv = curve_fit(two_lorentzian, zeta_in, spectral_corr, p0=params)
    elif len(params) == 7:
        popt, pconv = curve_fit(three_lorentzian, zeta_in, spectral_corr, p0=params)
    else:
        print('Wrong number of parameters!')
        print('4 for two lorentzians and 7 for three lorentzians!')
        return False
    return popt



# return the normalized spectral correlation of two lorentzians
def two_lorentzian(zeta_in, params):
    energy_vec = zeta_in

    # two lorentzians
    E, gamma, A, c = params
    lineshape = 1/(energy_vec**2 + 0.25 * gamma**2) + A/((energy_vec-E)**2 + 0.25 * gamma**2)

    # spectral correlation of the two lorentzians
    spectral_corr = np.correlate(lineshape, lineshape, 'full')
    spectral_corr = spectral_corr/max(spectral_corr) + c
    spectral_corr = spectral_corr/max(spectral_corr)
    return spectral_corr

# return the normalized spectral correlation of three lorentzians
def three_lorentzian(zeta_in, params):
    energy_vec = zeta_in

    # three lorentzians
    E0, E1, gamma, A0, A1, c, d = params
    lineshape = 1/(energy_vec**2 + 0.25 * gamma**2) + A0/((energy_vec-E0)**2 + 0.25 * gamma**2) + A1/((energy_vec-E1)**2 + 0.25 * gamma**2)

    # spectral correlation of the three lorentzians
    spectral_corr = np.correlate(lineshape, lineshape, 'full')
    spectral_corr = spectral_corr/max(spectral_corr) + c
    spectral_corr = d*spectral_corr/max(spectral_corr)
    return spectral_corr


'''
These functions are in the time domain.
'''

def fit_with_lorentzian_FFT(path_length_difference_in, interferogram, params):
    popt, pconv = curve_fit(lorentzian_FFT, path_length_difference_in, interferogram, p0=params)
    return popt


# return the Fourier transformed interferogram of the lorentzians
def lorentzian_FFT(path_length_difference, params):
    #some constants
    eV2cm = 8065.54429
    cm2eV = 1 / eV2cm

    # create a zeta_eV according to the input path_length_difference
    N =  4097 # number of grids we generate
    delta=(max(path_length_difference) - min(path_length_difference)) / (N-1)
    zeta_eV = np.fft.fftshift(np.fft.fftfreq(N, delta)) * cm2eV * 1000 # in meV

    # the spectral correlation using the params given on the zeta_eV
    if len(params) == 4:
        spectral_corr = two_lorentzian(zeta_eV, params)
    elif len(params) == 7:
        spectral_corr = three_lorentzian(zeta_eV, params)
    else:
        print('Wrong number of parameters!')
        print('4 for two lorentzians and 7 for three lorentzians!')
    return False

    # then FFT the spectral_corr to the Fourier domain
    interferogram = np.abs(np.fft.fftshift(np.fft.ifft(spectral_corr)))
    return interferogram



'''
The autocorrelation of a spectrum, which is the spectral correlation, is the Fourier Transform of the absolute square of the Fourier transform of the spectrum. Thus, the absolute squre of the Fourier transform of the spectrum is indeed the PCFS interferogram, in the time domain. So we only need to fit the absolute square of the Fourier transform of the spectrum to the PCFS interferogram.
'''

def lorentzian_FT(t, gamma, w):
    return np.exp(-t*w*1j - 0.5*gamma*np.abs(t))

def sum_lorentzian(ks, ws, gammas, cs):
    if len(ws) != len(gammas) or len(ws) != len(cs):
        raise ValueError('The length of gammas, ws, and cs should be equal')

    sum_lor = sum([c*0.5*gamma/np.pi/((ks-w)**2+0.25*gamma**2)
        for gamma, w, c in zip(gammas, ws, cs)])
    return sum_lor


def sq_fft_sum_lorentzian(t, ws, gammas, cs):
    if len(ws) != len(gammas) or len(ws) != len(cs):
        raise ValueError('The length of gammas, ws, and cs should be equal')

    FFT_sum_lor = sum([lorentzian_FT(t,gamma,w)*c for gamma, w, c in zip(gammas, ws, cs)])
    interferogram = np.abs(FFT_sum_lor)**2
    return interferogram


def get_params(u):
    if len(u)%3 != 0:
        raise ValueError("dimension of u must be a multiple of 3.")

    n = len(u) // 3
    ws = u[:n]
    gammas = u[n:2*n]
    cs = u[2*n:]

    return ws, gammas, cs


def fit_pcfs_wrapper(t, *u):
    ws, gammas, cs = get_params(u)

    return sq_fft_sum_lorentzian(t, ws, gammas, cs)


def do_fft(ys, dw):
    fys = np.fft.fft(ys) * dw
    ts = np.fft.fftfreq(ys.size, d=dw/(2*np.pi))
    idx = np.argsort(ts)

    return ts[idx], fys[idx]


if __name__ == '__main__':
    # ws = [0,4,6]
    # gammas = [1,0.3,0.2]
    # cs = [1,0.25,0.22]
    # t = np.arange(-12,12,0.2)
    #
    # ks = np.arange(-10,10,0.1)
    # dk = ks[1]-ks[0]
    # lor_f = sum_lorentzian(ks, ws, gammas, cs)
    # lor_f_ac = np.correlate(lor_f,lor_f,"same") * dk
    # ts, interferogram = do_fft(lor_f_ac, dk)
    # interferogram_fft = np.abs(interferogram)
    #
    # _, lor_t = do_fft(lor_f, dk)
    # interferogram_fft2 = np.abs(lor_t)**2.
    #
    # sq_fft_sum = sq_fft_sum_lorentzian(t,ws,gammas,cs)
    # plt.plot(ts, interferogram_fft, 'ro', markersize = 2)
    # plt.plot(ts, interferogram_fft2, 'go', markersize = 2)
    # plt.plot(t, sq_fft_sum, 'b-o', markersize = 2)
    # plt.xlim([-10,10])
    # plt.show()

    # -----------------------------------------------------------

    ts, ys = np.loadtxt("Dot1_run_2_interferogram.txt")
    ind = 10
    ts_fit = ts[ind:]
    ys_fit = ys[ind:]
    us = np.zeros(9)
    us[:3] = [-0.01, 0,0.01]
    res = curve_fit(fit_pcfs_wrapper, ts_fit, ys_fit, p0=us)
    ws, gammas, cs = get_params(res[0])
    print(ws)
    delta_w = np.diff(ws)
    delta_w = delta_w/2/np.pi*4.13567 # convert energy difference to meV
    print(delta_w)
    print(gammas)
    print(cs)
    print(res[1])

    plt.plot(ts[ind:], sq_fft_sum_lorentzian(ts[ind:],ws,gammas,cs), "-")
    plt.plot(ts, ys, 'x')
    plt.xlim([0,30])
    plt.xlabel('Path length difference [ps]')
    plt.ylabel(r'$g^{(2)}_{cross} - g^2_{auto}$')
    plt.legend([r'Fitted with three lorentzians $\Delta = 0.6635, 0.002$ meV', 'Raw data'])
    plt.title('Dot1_run_two PCFS interferogram averaged over 50-200 ms')
    plt.show()

    # -----------------------------------------------------------
    #
    # scales = [0.7, 1.5, 3.7]
    # phases = [0.5, 1.7, 3.2]
    # ts_posit = np.concatenate([np.random.exponential(scale, 10000) for scale in scales])
    # ts_negat = np.concatenate([-np.random.exponential(scale, 10000) for scale in scales])
    # ts = np.concatenate([ts_posit, ts_negat])
    # hist, bin_edges = np.histogram(ts, bins=1024)
    # # plt.semilogy(bin_edges[:-1], hist, "x")
    # # plt.show()
    #
    # dbin = bin_edges[1] - bin_edges[0]
    # fs, hist_f = do_fft(hist, dbin)
    #
    # df = fs[1] - fs[0]
    # hist_f_ac = np.correlate(hist_f,hist_f,"same") * df
    # ts2, interferogram = do_fft(hist_f_ac, df)
    # plt.plot(ts2, np.abs(interferogram))
    # plt.show()
