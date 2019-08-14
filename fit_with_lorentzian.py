'''
This function fits the spectral correlation to multiple lorentzians.
'''

import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt



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

 
