import numpy as np
import matplotlib.pyplot as plt
from scipy.special import i0, i1    # modified bessel function
from scipy.special import exp1
from ssp_library import *

np.seterr(invalid='ignore')

def estimate_noise_psd(noise_psd, pow_spectrum, i_frame):
    return (noise_psd*i_frame + pow_spectrum) / (i_frame+1)

def imcra(y_PSD):
    """
    :param y_PSD: noisy power spectral density
    :return: noise estimate
    """
    n_frm, n_freq = y_PSD.shape
    # wsub = np.asarray([[0.25],[0.5],[0.25]])
    wsub = np.asarray([0.25, 0.5, 0.25])
    n_PSD = np.zeros([n_frm, n_freq])

    u = 8       # subwindow
    v = 15      # ~~
    d = u * v

    alpha = 0.92             #
    alpha_s = 0.9            #
    alpha_d = 0.85           #
    beta = 1.47              #
    Bmin = 1.66              #
    zeta_0 = 1.67            #
    gamma_0 = 4.6            #
    gamma_1 = 3              #
    gamma_2 = gamma_1 - 1    # ??

    sapmax = 0.95            # ??

    s_hat_sub_min = np.zeros([u, n_freq])
    s_bar_sub_min = np.zeros([u, n_freq])

    ## initialization
    n_PSD[1, :] = y_PSD[1, :]
    n_PSD[2, :] = y_PSD[1, :]
    s_f = np.convolve(y_PSD[1, :],wsub,'same')
    s_hat = s_f
    s_bar = s_f
    s_hat_sub = s_f
    s_bar_sub = s_f

    for j_frm in range(u):
        s_hat_sub_min[j_frm, :] = s_f
        s_bar_sub_min[j_frm, :] = s_f

    s_hat_min = s_f
    s_bar_min = s_f

    gain_i_frm = np.ones(n_freq)       ## 선언해야 하는 이유?
    gamma_prev = np.ones(n_freq)
    n_PSD_bias = n_PSD[1, :]

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
        # print(s_hat.shape)
        # print(s_hat_min.shape)
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
            # print('--')
            # print(s_hat_sub_min.shape)
            # print('--')
            s_hat_min               = np.amin(s_hat_sub_min, axis = 0)  # 1 -> 0
            s_bar_sub_min[1:u, :] = s_bar_sub_min[:u-1, :]
            s_bar_min               = np.amin(s_bar_sub_min, axis = 0)

        gamma_prev = gamma

    return n_PSD


def spectral_subtraction(pow_spectrum, noise_psd, min_gain):
    """
     pow_spectrum: power spectrum of input noisy signal
     noise_psd:    noise power spectral density
     min_gain:     minimum gain threshold
     return:       spectral subtraction gain
    """
    gain = 1 - np.sqrt(noise_psd / (pow_spectrum + 1e-10))
    return np.maximum(gain, min_gain)

def wiener(pow_spectrum, noise_psd, prev_xi, alpha, min_gain):
    """
     pow_spectrum: power spectrum of input noisy signal
     noise_psd:    noise power spectral density
     prev_xi:      buffer to compute a priori SNR recursively (decision-directed approach)
     alpha:        first order coefficient for decision-directed approach
     min_gain:     minimum gain threshold
     return:       Wiener gain
    """
    # Compute a posteriori SNR.
    gamma = pow_spectrum / (noise_psd + 1e-6)

    # Compute a priori SNR using a decision-directed approach.
    xi = alpha * prev_xi + (1 - alpha) * np.maximum(gamma-1, 0)

    # Compute Wiener gain.
    gain = np.maximum(xi / (1 + xi), min_gain)

    # Update a priori SNR for the next frame processing.
    prev_xi = gain * gain * gamma

    return prev_xi, gain
def Power_sub(pow_spectrum, noise_psd, prev_xi, alpha, min_gain):
    """
     pow_spectrum: power spectrum of input noisy signal
     noise_psd:    noise power spectral density
     prev_xi:      buffer to compute a priori SNR recursively (decision-directed approach)
     alpha:        first order coefficient for decision-directed approach
     min_gain:     minimum gain threshold
     return:       Wiener gain
    """
    # Compute a posteriori SNR.
    gamma = pow_spectrum / (noise_psd + 1e-6)

    # Compute a priori SNR using a decision-directed approach.
    xi = alpha * prev_xi + (1 - alpha) * np.maximum(gamma-1, 0)

    # Compute Wiener gain.
    gain = np.maximum(np.sqrt(xi / (1 + xi)), min_gain)

    # Update a priori SNR for the next frame processing.
    prev_xi = gain * gain * gamma

    return prev_xi, gain

def ml(pow_spectrum, noise_psd, min_gain):
    est_speech_pow_spectrum = np.maximum(pow_spectrum - noise_psd, 0)
    gain = 0.5 + 0.5 * np.sqrt(est_speech_pow_spectrum / (pow_spectrum + 1e-10))
    return np.maximum(gain, min_gain)

def ml_soft_decision(pow_spectrum, noise_psd, min_gain):
    est_speech_pow_spectrum = np.maximum(pow_spectrum - noise_psd, 0)
    gain1 = 0.5 + 0.5*np.sqrt(est_speech_pow_spectrum / (pow_spectrum + 1e-10))

    gamma = pow_spectrum / (noise_psd + 1e-10)
    xi = gamma - 1
    i = np.minimum(2*np.sqrt(xi*est_speech_pow_spectrum / (noise_psd + 1e-10)), 400)
    up = np.exp(-xi)*i0(i)
    down = 1 + up + 1e-10
    speech_presence_prob = up / down
    gain2 = speech_presence_prob
    # speech_presence_prob_down = 1 + speech_presence_prob_up
    # gain2 = speech_presence_prob_up / (speech_presence_prob_down + 1e-10)
    gain = gain1*gain2

    return gain

def ml_sd(pow_spectrum, noise_psd, min_gain, snr, q):
    """
    param pow_spectrum: noisy power spectral density
    param noise_psd:    noise power spectral density
    param min_gain:     minimum gain threshold
    param snr :         Signal to Noise Ratio (SNR)
    param q :           speech absence probability
    return:             Maximum Likelihood Envelope gain
    """
    xi = 10 ** ( snr / 10 ) #
    # Compute A
    A = np.maximum(0.5 * (1 + np.sqrt(np.maximum(pow_spectrum - noise_psd, 0) / pow_spectrum)), min_gain)  # max 값: 뺀값과 0 중에 큰 값을 쓰겠다. -> 최솟값이 0
    # Compute P(H1|V)
    p = np.minimum(2 * np.sqrt(xi * pow_spectrum / noise_psd), 400)
    P = np.exp(-xi) * i0(p)
    Q = (q * P) / (1 + (q * P))
    gain = A*Q
    return gain

# def ml(y_PSD, n_PSD, delta, snr, q):
#     """
#     :param y_PSD: noisy power spectral density
#     :param n_PSD: noise power spectral density
#     :param delta: minimum gain threshold
#     :param snr : Signal to Noise Ratio (SNR)
#     :param q : speech absence probability
#     :return: Maximum Likelihood Envelope gain
#     """
#     xi = 10 ** ( snr / 10 )
#     G = np.maximum(0.5 * (1 + np.sqrt(np.maximum(y_PSD - n_PSD,0) / y_PSD)), delta);
#
#     p = np.minimum(2 * np.sqrt(xi * y_PSD / n_PSD), 400)
#     P = np.exp(-xi) * i0(p)
#     Q = (q * P) / (1 + (q * P))
#
#     return G * Q

def mmse_stsa(pow_spectrum, noise_psd, prev_xi, alpha, sap, min_gain):
    """
     pow_spectrum: power spectrum of input noisy signal
     noise_psd:    noise power spectral density
     prev_xi:      buffer to compute a priori SNR recursively (decision-directed approach)
     alpha:        first order coefficient for decision-directed approach
     sap:          speech absence probability
     min_gain:     minimum gain threshold
     return:       MMSE-STSA gain
    """
    const_gamma = np.sqrt(np.pi) / 2        # gamma(1.5) = sqrt(pi)/2
    mu = (1 - sap) / sap

    # Compute a posteriori SNR.
    gamma = pow_spectrum / (noise_psd + 1e-6)

    # Compute a priori SNR using a decision-directed approach.
    xi = alpha * prev_xi + (1 - alpha) * np.maximum(gamma-1, 0)

    # Compute nu and eta values.
    # if nu is too big, lambda_a will be exploded. Thus, limit the value here.
    nu = np.minimum(xi / (1 + xi) * gamma, 600)
    eta = xi / (1 - sap)

    # Compute gain by dividing the equation into three parts.
    lambda_a = mu * np.exp(nu) / (1 + eta)
    gain_m = lambda_a / (1 + lambda_a)
    gain_1 = (1 + nu) * i0(nu / 2) + nu * i1(nu / 2)
    gain_2 = const_gamma * np.sqrt(nu) / (gamma + 1e-6) * np.exp(-nu / 2) * gain_1
    gain_final = gain_m * gain_2

    gain = np.maximum(gain_final, min_gain)

    # Update a priori SNR for the next frame processing.
    prev_xi = gain * gain * gamma

    return prev_xi, gain
# def plot_graph(noise_psd, pow_spectrum, gain, Wiener, Power_sub, ml):


def mmse_lsa(pow_spectrum, noise_psd, prev_xi, alpha, min_gain):
    gamma = pow_spectrum / (noise_psd + 1e-6)
    xi = alpha * prev_xi + (1 - alpha) * np.maximum(gamma - 1, 0)
    nu = np.minimum(xi / (1 + xi) * gamma, 600)

    gain = np.maximum(xi / (1 + xi) * np.exp(exp1(nu)/2), min_gain)

    prev_xi = gain * gain * gamma

    return prev_xi, gain

def om_lsa(y_PSD, n_PSD, delta, alpha):
    """
    :param y_PSD: noisy power spectral density
    :param n_PSD: noise power spectral density
    :param delta: minimum gain threshold
    :param alpha: alpha in DD approach
    :return: om-lsa gain
    """
    wlocal = 1
    wglobal = 15

    hlocal = hann_window(2 * wlocal + 1)
    hlocal = hlocal / sum(hlocal)              # local 평균값
    hglobal = hann_window(2 * wglobal + 1)
    hglobal = hglobal / sum(hglobal)           # global

    zetamin = 10 ** (-10 / 10)                 # -10dB
    zetamax = 10 ** (-5 / 10)                  # -5dB
    zetaratio = np.log(zetamax / zetamin)      # mu 값 결정에 사용

    zetapmin = 10 ** (0 / 10)                  # 0dB
    zetapmax = 10 ** (10 / 10)                 # 10dB

    sapmax = 0.95
    beta = 0.7

    n_frm, n_freq = y_PSD.shape
    gain = np.zeros([n_frm, n_freq])
    xi_old = 1 ## if i_frm == 0 -> xi = alpha + (1 - alpha) * np.maximum(gamma-1, 0)
    for i_frm in range(n_frm):
        if i_frm == 0:
            npsd = y_PSD[i_frm, :]             # i_frm == 0일 때 n_PSD 값 256개가 모두 0이어서 noisy인 y_PSD를 그대로 가져옴
        else:
            npsd = n_PSD[i_frm, :]
        # npsd = n_PSD[i_frm, :]
        ypsd = y_PSD[i_frm, :]
        # print(npsd)
        # if i_frm == 0:
        #     print(ypsd)


        gamma = ypsd / npsd
        xi = alpha * xi_old + (1 - alpha) * np.maximum(gamma-1, 0)
        nu = xi / (1 + xi) * gamma

        if i_frm == 0:
            zeta = xi
            zetaframe = np.mean(zeta)
            zetapeak = zetaframe
            sap = sapmax * np.ones(n_freq)
        else:
            zeta = beta * zetaprev + (1 - beta) * xi_prev
            zetalocal = np.convolve(zeta, hlocal,'same')
            zetaglobal = np.convolve(zeta, hglobal, 'same')
            zetaframe = np.mean(zeta)

            mu = np.maximum(np.minimum(np.log(zetaframe / zetapeak / zetamin) / zetaratio, 1), 0)
            Plocal = np.maximum(np.minimum(np.log(zetalocal / zetamin) / zetaratio, 1), 0)
            Pglobal = np.maximum(np.minimum(np.log(zetaglobal / zetamin) / zetaratio, 1), 0)
            if zetaframe > zetamin:
                if zetaframe > zetaframeprev:
                    Pframe = 1
                    zetapeak = np.minimum(np.maximum(zetaframe, zetapmin), zetapmax)
                else:
                    Pframe = mu
            else:
                Pframe = 0

            sap = np.minimum(1 - Pframe * Plocal * Pglobal, sapmax)

        spp = 1 / (1 + sap /(1 - sap) * (1 + xi) * np.exp(-nu))

        ## gain estimation
        gain_i_frm = xi / (1 + xi) * np.exp(exp1(nu)/2)
        gain[i_frm, :] = np.minimum((gain_i_frm ** spp) * (delta ** (1 - spp)) ,1)   # delta: minimum gain threshold

        ##finalization
        xi_old = (gain_i_frm ** 2) * gamma
        xi_prev = xi
        zetaframeprev = zetaframe
        zetaprev = zeta

    return gain

def plot_waveform(clean_signal, input_signal, enhanced_signal, fs):
    n_sample = min(list((np.size(clean_signal), np.size(input_signal), np.size(enhanced_signal))))
    max_Y = 1.2 * max(list((max(abs(clean_signal)), max(abs(input_signal)), max(abs(enhanced_signal)))))
    t = n_sample / fs
    n = np.linspace(0, t, n_sample)

    plt.figure(1)
    plt.subplot(311)
    plt.plot(n, clean_signal[:n_sample], color='black')
    plt.title('clean, noisy, enhanced')
    plt.ylabel('Amplitude')
    plt.axis([0, t, -max_Y, max_Y])
    plt.subplot(312)
    plt.plot(n, input_signal[:n_sample], color='black')
    plt.ylabel('Amplitude')
    plt.axis([0, t, -max_Y, max_Y])
    plt.subplot(313)
    plt.plot(n, enhanced_signal[:n_sample], color='black')
    plt.ylabel('Amplitude')
    plt.axis([0, t, -max_Y, max_Y])
    plt.xlabel('Time (s)')
    plt.show()

def plot_spectrogram(clean_signal, input_signal, enhanced_signal, fs):
    frm_size = 0.032
    ratio = 0.9
    n_fft = int(fs * frm_size)
    n_overlap = int(fs * frm_size * ratio)
    win = np.hanning(n_fft)

    plt.figure(2)
    # plt.subplot(311)
    # plt.specgram(clean_signal, NFFT=n_fft, Fs=int(fs), window=win, noverlap=n_overlap, cmap='jet', vmin=-30, vmax=60)
    plt.specgram(clean_signal, NFFT=n_fft, Fs=int(fs), window=win, noverlap=n_overlap, cmap='jet', vmin=-15, vmax=40)
    plt.colorbar()
    plt.title('Clean signal')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.show()
    # plt.subplot(312)

    plt.specgram(input_signal, NFFT=n_fft, Fs = int(fs), window=win, noverlap=n_overlap, cmap='jet', vmin=-15, vmax=40)
    plt.colorbar()
    plt.title('Noisy signal')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.show()
    # plt.subplot(313)

    plt.specgram(enhanced_signal, NFFT=n_fft, Fs=int(fs), window=win, noverlap=n_overlap, cmap='jet', vmin=-15, vmax=40)
    plt.colorbar()
    plt.title('Enhanced signal')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.show()
