import numpy as np
import scipy.io.wavfile as wav
from ssp_library import *
from scipy.special import exp1
np.seterr(invalid='ignore')


def average_noise_PSD(y_spec, k):
    """
    :param y_spec: noisy spectrum
    :param k: averaging frame range (1:K)
    :return: average noise PSD
    """
    n_frm, n_freq = y_spec.shape
    N = np.zeros([n_frm, n_freq])
    ave_N = np.mean(abs(y_spec[0:k])**2, 0)

    for i_frm in range(n_frm):
        N[i_frm] = ave_N
    return N




def imcra(y_PSD):
    """
    :param y_PSD: noisy power spectral density
    :return: noise estimate
    """
    n_frm, n_freq = y_PSD.shape
    wsub = np.asarray([0.25, 0.5, 0.25])
    n_PSD = np.zeros([n_frm, n_freq])

    u = 8       # subwindow
    v = 15      # ~~
    d = u * v

    alpha = 0.92
    alpha_s = 0.9
    alpha_d = 0.85
    beta = 1.47
    Bmin = 1.66
    zeta_0 = 1.67
    gamma_0 = 4.6
    gamma_1 = 3
    gamma_2 = gamma_1 - 1

    sapmax = 0.95

    s_hat_sub_min = np.zeros([u, n_freq])
    s_bar_sub_min = np.zeros([u, n_freq])

    ## initialization
    n_PSD[0, :] = y_PSD[0, :]
    n_PSD[1, :] = y_PSD[0, :]
    s_f = np.convolve(y_PSD[0, :],wsub,'same')
    s_hat = s_f
    s_bar = s_f
    s_hat_sub = s_f
    s_bar_sub = s_f

    for j_frm in range(u):
        s_hat_sub_min[j_frm, :] = s_f
        s_bar_sub_min[j_frm, :] = s_f

    s_hat_min = s_f
    s_bar_min = s_f

    gain_i_frm = np.ones(n_freq)
    gamma_prev = np.ones(n_freq)
    n_PSD_bias = n_PSD[0, :]

    ## noise estimation
    for i_frm in range(1, n_frm - 1):
        ind = i_frm % v + 1
        s_f = np.convolve(y_PSD[i_frm, :], wsub, 'same')

        ## SNR related params
        gamma      = y_PSD[i_frm, :] / n_PSD[i_frm, :]
        xi         = alpha * (gain_i_frm ** 2) * gamma_prev + (1 - alpha) * np.maximum(gamma-1, 0)
        nu         = gamma * xi / (1 + xi)
        gain_i_frm = xi / (1 + xi) * np.exp(exp1(nu) / 2)

        ## first smoothing
        s_hat = alpha_s * s_hat + (1 - alpha_s) * s_f

        ## first minimum
        if ind == 1:
            s_hat_sub = s_hat
        else:
            s_hat_sub = np.minimum(s_hat,s_hat_sub)

        s_hat_min           = np.minimum(s_hat, s_hat_min)
        s_hat_sub_min[1, :] = s_hat_sub

        ## rough vad
        vad_hat   = np.zeros(n_freq)
        gamma_hat = y_PSD[i_frm, :] / s_hat_min / Bmin
        zeta_hat  = s_hat / s_hat_min / Bmin
        for i_freq in range(n_freq):
            if gamma_hat[i_freq] < gamma_0 and zeta_hat[i_freq] < zeta_0:
                vad_hat[i_freq] = 1

        ## second smoothing
        num = np.convolve(y_PSD[i_frm, :] * vad_hat, wsub, 'same')
        den = np.convolve(vad_hat, wsub, 'same')
        s_bar_f = num / den
        for i_freq in range(n_freq):
            if den[i_freq] == 0:
                s_bar_f = s_bar
        s_bar = alpha_s * s_bar + (1 - alpha_s) * s_bar_f

        ## second minimum
        if ind == 1:
            s_bar_sub = s_bar
        else:
            s_bar_sub = np.minimum(s_bar, s_bar_sub)

        s_bar_min = np.minimum(s_bar, s_bar_min)
        s_bar_sub_min[1, :] = s_bar_sub

        ## speech absence/presence probability
        gamma_bar = y_PSD[i_frm, :] / s_bar_min / Bmin
        zeta_bar  = s_hat / s_bar_min / Bmin
        sap       = (gamma_1 - gamma_bar) / gamma_2
        for i_freq in range(n_freq):
            if zeta_bar[i_freq] >= zeta_0:
                sap[i_freq] = 0
        sap = np.minimum(np.maximum(sap, 0), sapmax)
        spp = 1 / (1 + sap / (1 - sap) * (1 + xi) * np.exp(-nu))

        ## noise PSD estimation
        alpha_tilde_d     = alpha_d + (1 - alpha_d) * spp
        n_PSD_bias        = alpha_tilde_d * n_PSD_bias + (1 - alpha_tilde_d) * y_PSD[i_frm, :]
        n_PSD[i_frm+1, :] = beta * n_PSD_bias

        ## finalization
        if ind == v:
            s_hat_sub_min[1:u, :] = s_hat_sub_min[:u-1, :]
            s_hat_min               = np.amin(s_hat_sub_min, axis = 0)  # 1 -> 0
            s_bar_sub_min[1:u, :] = s_bar_sub_min[:u-1, :]
            s_bar_min               = np.amin(s_bar_sub_min, axis = 0)

        gamma_prev = gamma

    return n_PSD