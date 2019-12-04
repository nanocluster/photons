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


def do_fft(ys, dw):
    fys = np.fft.fft(ys) * dw
    ts = np.fft.fftfreq(ys.size, d=dw/(2*np.pi))
    idx = np.argsort(ts)

    return ts[idx], fys[idx]

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

    if len(u)%3 != 2:
        raise ValueError("caomei zhen de huai!")

    # [omegas, gammas, cs]
    u = np.abs(u)
    u = np.concatenate([[0], u])
    n = len(u) // 3
    ws = u[:n]
    gammas = u[n:2*n]
    cs = u[2*n:]
    return ws, gammas, cs


def fit_pcfs_wrapper(t, *u):
    ws, gammas, cs = get_params(u)

    return sq_fft_sum_lorentzian(t, ws, gammas, cs)



def monoexp(x, a, b):
    return b*np.exp(a*x)

def find_tau(x,y):
    ind = np.argmin(abs(y-np.exp(-1)))
    return x[ind]

def square(t,*u):
    n = int(len(u)//3)
    gammas = np.abs(u[:n])
    ws = u[n:2*n]
    cs = np.abs(u[2*n:])
    FFT_sum_lor = sum([lorentzian_FT(t,gamma,w)*c for gamma, w, c in zip(gammas, ws, cs)])
    interferogram = np.abs(FFT_sum_lor)**2
    return interferogram

if __name__ == '__main__':

    path = 'C:\\Users\\weiwei\\Downloads\\DotE_run_one'
    dotID = path.split('\\')[-1]
    ts = np.loadtxt(path+'\\path_length_time.dat')
    ys = np.loadtxt(path+'\\interferogram.dat')
    ind = 7
    ind_end = -10
    ts_fit = ts[ind:ind_end]
    ys_fit = ys[ind:ind_end]
    # plt.plot(ts_fit,ys_fit,'-x')
    # plt.show()


    nsidepeak = 2
    us = np.zeros(3*nsidepeak+2)
    us[:nsidepeak] = [0.5,1]
    us[nsidepeak:-nsidepeak-1]=[0.04,0.04,3]
    us[-nsidepeak-1:] = [0.1,0.1,0.1]
    res = curve_fit(fit_pcfs_wrapper, ts_fit, ys_fit, p0=us)
    ws, gammas, cs = get_params(res[0])
    print('ws', ws)
    delta_w = ws[1:]-ws[0]
    delta_w = delta_w/2/np.pi*4.13567 # convert energy difference to meV
    print('delta w', delta_w)
    print('gamma', gammas)
    print('tau',1/gammas[0])
    print('c', cs)
    print('res', res[1])


    # grid = np.linspace(0,100,1000)
    # ys_total = sq_fft_sum_lorentzian(grid,ws,gammas,cs)
    # ys_coherent = ys_total
    # ys_ind = ys_coherent[1:-1]
    # ts_ind = grid[1:-1]
    # ys_after = ys_ind-ys_coherent[2:]
    # ys_before = ys_ind-ys_coherent[:-2]
    # ts_decay = ts_ind[(ys_after>0)*(ys_before>0)]
    # # ts_decay = np.concatenate([[0],ts_decay])
    # ys_decay = sq_fft_sum_lorentzian(ts_decay,ws,gammas,cs)
    # plt.semilogy(ts_decay,ys_decay,'x')
    # plt.semilogy(ts[ind:], sq_fft_sum_lorentzian(ts[ind:],ws,gammas,cs), "-", c='orange')
    # plt.show()
    #
    # decay_p = curve_fit(monoexp, ts_decay,ys_decay,[-0.2,1])[0]



    # ind = 3
    plt.plot(ts[ind:], sq_fft_sum_lorentzian(ts[ind:],ws,gammas,cs), "-",c = 'orange',lw = 2, label = r'Fitted with three lorentzians $\Delta =$'+str(delta_w[0])[:5]+' mev')#+' and '+str(delta_w[1])[:6]+' meV')
    plt.plot(ts[ind:], ys[ind:], 'x',c = 'grey',label = 'Raw data')
    # plt.plot(ts[ind:],monoexp(ts[ind:],*decay_p), '--',c = 'r',label = r'Envelope decay $T_2/2 = $' + str(-1/decay_p[0])[:4]+' ps')
    # plt.xlim([0,60])
    plt.xlabel('Path length difference [ps]')
    plt.ylabel(r'$g^{(2)}_{cross} - g^2_{auto}$')
    plt.legend()
    plt.title(dotID+' PCFS interferogram averaged')
    plt.show()

    ks = np.linspace(-2.5,2.5,2**11)
    y = sum_lorentzian(ks,ws,gammas,cs)/max(sum_lorentzian(ks,ws,gammas,cs))
    y =np.abs(np.fft.fftshift(np.fft.fft(np.abs(np.fft.fft(y))**2)))
    y = y/max(y)
    x = ks/2/np.pi*4.13567 # convert energy difference to meV
    plt.plot(x,y)
    plt.ylabel(r'Normalized $p(\zeta)$')
    plt.xlabel(r'$\zeta$ [meV]')
    plt.title(dotID+' Fitted Spectral Correlation')
    plt.show()


    # # play in the spectral domain
    #
    # mirror_interf = np.concatenate([ys[::-1][:-ind-1],ys[ind:]])
    # mirror_time = np.concatenate([-ts[::-1][:-ind-1],ts[ind:]])
    # mirror_stage_pos = mirror_time*2.997/100 # in cm
    # n = len(mirror_stage_pos)
    # delta = (max(mirror_stage_pos)-min(mirror_stage_pos))/(n-1)
    # ks = np.linspace(min(mirror_stage_pos),max(mirror_stage_pos),n)
    # #some constants
    # eV2cm = 8065.54429
    # cm2eV = 1 / eV2cm
    # zeta_eV = np.fft.fftshift(np.fft.fftfreq(n, delta)) * cm2eV * 1000 # in meV
    # spectral_correlation = np.abs(np.fft.fftshift(np.fft.fft(mirror_interf)))
    # spectral_correlation = spectral_correlation/max(spectral_correlation)
    # y = sum_lorentzian(ks,ws,gammas,cs)/max(sum_lorentzian(ks,ws,gammas,cs))
    # y =np.abs(np.fft.fftshift(np.fft.fft(np.abs(np.fft.fft(y))**2)))
    # y = y/max(y)
    # x = ks/2/np.pi*4.13567
    # plt.plot(zeta_eV,spectral_correlation,'-')
    # # plt.plot(x,y,'r')
    # plt.xlim(-2,2)
    # plt.ylabel(r'Normalized $p(\zeta)$')
    # plt.xlabel(r'$\zeta$ [meV]')
    # plt.title(dotID+' Spectral Correlation')
    # plt.show()
    # plt.show()
    # u = [0,0.1,1,0.06,0.06,1,1,0.1,0.1]
    # n = 3
    # us = curve_fit(square,zeta_eV,spectral_correlation,p0=u)[0]
    # gammas = us[:n]
    # ws = us[n:2*n]
    # cs = us[2*n:]
    # print('ws',ws)
    # delta_w = ws[1:]-ws[0]
    # delta_w = delta_w/2/np.pi*4.13567 # convert energy difference to meV
    # print('delta w', delta_w)
    # print('gamma', gammas)
    # print('c', cs)
    # print('res', res[1])
    # plt.plot(zeta_eV,square(zeta_eV,*us))
    # plt.plot(zeta_eV,spectral_correlation)
    # plt.show()
