import numpy as np
import scipy.io.wavfile as wav
from scipy.special import exp1
from ssp_library import *
import mpmath
import matplotlib.pyplot as plt
from gain_estimation_advanced import *
from spectrogram import *

def imcra_test(y_PSD):
    """
    :param y_PSD: noisy power spectral density
    :return: noise estimate
    """
    n_frm, n_freq = y_PSD.shape
    wsub = np.asarray([0.25, 0.5, 0.25])                # 1x3
    n_PSD = np.zeros([n_frm, n_freq])

    u = 8                   # sub window
    v = 15                  # number of sample
    D = u * v               # size of the minimum search window

    alpha = 0.92            # xi estimation 에 사용되는 weighting factor - om lsa
    alpha_s = 0.9           # smoothing parameter
    alpha_d = 0.85          # time-varying frequency-dependent smoothing parameter: minimum 값으로 이용되는거라 생각
    beta = 1.47             # compensation factor
    Bmin = 1.66             # bias of a minimum noise estimate
    zeta_0 = 1.67           # initialize
    gamma_0 = 4.6           # initialize
    gamma_1 = 3             # initialize

    sapmax = 0.95           # threshold?

    s_hat_sub_min = np.zeros([u, n_freq])
    s_bar_sub_min = np.zeros([u, n_freq])

    """initialization"""
    n_PSD[0, :] = y_PSD[0, :]                         # y_psd 0번째를 noise psd 0번째와 1번째로 초기화
    n_PSD[1, :] = y_PSD[0, :]
    s_f = np.convolve(y_PSD[0, :], wsub, 'same')      # 0번째 프레임에 대한 Sf - initialize
    s_hat = s_f
    s_bar = s_f
    s_hat_sub = s_f
    s_bar_sub = s_f

    for j_frm in range(u):                  # S_hat_min 값을 u 개의 sub window 내에서 구하겠다??
        s_hat_sub_min[j_frm, :] = s_f
        s_bar_sub_min[j_frm, :] = s_f

    s_hat_min = s_f
    s_bar_min = s_f

    Gi = np.ones(n_freq)                # xi 만드는데 이용되는 gain i
    gamma_prev = np.ones(n_freq)
    n_PSD_bias = n_PSD[0, :]            # final function에 이용되는 n_PSD

    """noise estimation"""
    for i_frm in range(1, n_frm - 1):                       # 첫번째와 마지막 프레임은 제외하고 for 문 돌리겠다.
        ind = i_frm % v + 1                                 # v의 크기만큼이 되면 갱신시켜주기 위한 부분
        s_f = np.convolve(y_PSD[i_frm, :], wsub, 'same')    # smoothing in frequency domain
                                                            # b(i): wsub의 내에서 합이 1인  nomalized widow function
        """SNR related params"""
        gamma = y_PSD[i_frm, :] / n_PSD[i_frm, :]           # a posteriori SNR = noisy_P / noise_P
        xi = alpha * (Gi ** 2) * gamma_prev + (1 - alpha) * np.maximum(gamma - 1, 0)    # Decision-directed, modified version - om lsa
        nu = gamma * xi / (1 + xi)                          # nu 정의
        Gi = xi / (1 + xi) * np.exp(exp1(nu) / 2)           # Gi: spectral gain function - om lsa

        """first smoothing"""
        s_hat = alpha_s * s_hat + (1 - alpha_s) * s_f       # recursive averaging in time domain smoothing

        """first minimum"""
        if ind == 1:                                    # v 개의 sample 이 새롭게 시작될떄
            s_hat_sub = s_hat                           # 이전의 noisy power 사용
        else:
            s_hat_sub = np.minimum(s_hat, s_hat_sub)    # 그 이후는 min 값 비교하여 power 로 사용
        # print(s_hat.shape)
        # print(s_hat_min.shape)
        s_hat_min = np.minimum(s_hat, s_hat_min)        # 위 과정을 통해  Smin 값 정해주기 (first smoothing 된 power vs noisy power)
        s_hat_sub_min[0, :] = s_hat_sub

        """rough vad"""
        vad_hat = np.zeros(n_freq)                      # vad 를 위해 모든 위치에 0
        gamma_hat = y_PSD[i_frm, :] / s_hat_min / Bmin
        zeta_hat = s_hat / s_hat_min / Bmin
        for i_freq in range(n_freq):                    # vad = 1: Speech absence
            if gamma_hat[i_freq] < gamma_0 and zeta_hat[i_freq] < zeta_0:
                vad_hat[i_freq] = 1
        # plt.plot(vad_hat)
        # plt.show()

        """second smoothing"""
        num = np.convolve(y_PSD[i_frm, :] * vad_hat, wsub, 'same')      # vad에 따라 b(i) 만큼의 power convolution
        den = np.convolve(vad_hat, wsub, 'same')                        # vad값을 b(i) 만큼 convolution

        s_bar_f = num / den                                             # smoothing in frequency domain
        for i_freq in range(n_freq):
            if den[i_freq] == 0:                                    # if vad 합이 = 0 이면: speech presence
                s_bar_f = s_bar                                     # 이전값 넣어주기
                                                                    # otherwise: speech absence
        s_bar = alpha_s * s_bar + (1 - alpha_s) * s_bar_f           # smoothing in time domain

        """second minimum"""
        if ind == 1:                         # first minimum 과 같은 과정
            s_bar_sub = s_bar
        else:
            s_bar_sub = np.minimum(s_bar, s_bar_sub)

        s_bar_min = np.minimum(s_bar, s_bar_min)            # first, second smoothing 된 power vs noisy power
        s_bar_sub_min[0, :] = s_bar_sub

        """speech absence/presence probability"""
        gamma_bar = y_PSD[i_frm, :] / s_bar_min / Bmin
        zeta_bar = s_hat / s_bar_min / Bmin
        # SAP
        sap = (gamma_1 - gamma_bar) / (gamma_1 - 1)                 # if, 1 < gamma_var < gamma_1 & zeta_bar < zeta_0
        for i_freq in range(n_freq):
            if zeta_bar[i_freq] >= zeta_0:                          # otherwise 부분
                sap[i_freq] = 0
        sap = np.minimum(np.maximum(sap, 0), sapmax)                # sap 의 threshold 설정
        # SPP
        spp = 1 / (1 + sap / (1 - sap) * (1 + xi) * np.exp(-nu))    # 위에서 구한 parameter + sap 이용해서 spp 도출


        """noise PSD estimation"""
        alpha_tilde_d = alpha_d + (1 - alpha_d) * spp                                       # Time-varying smoothing parameter
        n_PSD_bias = alpha_tilde_d * n_PSD_bias + (1 - alpha_tilde_d) * y_PSD[i_frm, :]     # Compute noise power
        # plt.plot(alpha_tilde_d)
        # plt.show()

        # Final function
        n_PSD[i_frm + 1, :] = beta * n_PSD_bias                 # Multiplying bias compensation factor
        # plt.imshow(n_PSD)
        # plt.show()

        """finalization"""
        if ind == v:
            s_hat_sub_min[1:u, :] = s_hat_sub_min[:u - 1, :]        # v sample 이 될때마다 u sub window 내에서 minimum 값이 결정되고 저장된다.
            s_hat_min = np.amin(s_hat_sub_min, axis=0)  # 1 -> 0
            s_bar_sub_min[1:u, :] = s_bar_sub_min[:u - 1, :]
            s_bar_min = np.amin(s_bar_sub_min, axis=0)

        gamma_prev = gamma

    return n_PSD


fs, signal = wav.read('DR1_FAKS0_SA1_pink_0dB.wav')
signal = signal/32768
frm_size = 0.032
ratio = 0.5
delta = 0.03
alpha = 0.92
sap = 0.5
k = 5
win = hann_window(int(fs * frm_size))
stft_Y = stft(signal, fs, win, frm_size, ratio)
n_PSD = imcra_test(abs(stft_Y) ** 2)

# plot_spectrogram_1(n_PSD, fs)
# plt.imshow(n_PSD)
# plt.plot(noise)
# plt.imshow(noise)
# plt.show()

print(n_PSD.shape)
print(stft_Y.shape)
length_waveform = int(fs * frm_size) + int((fs * frm_size * ratio)*(len(stft_Y) - 1))

# gain = wiener(abs(stft_Y) ** 2, n_PSD, delta)
# gain = mmse_stsa(abs(stft_Y) **2, n_PSD, delta, alpha, sap)
gain = om_lsa(abs(stft_Y)**2, n_PSD, delta, alpha)
enhanced_S = stft_Y * gain
istft_enhanced_S = istft(enhanced_S, win, win, length_waveform, ratio)*32768
istft_enhanced_S = istft_enhanced_S.astype(np.int16)

plot_spectrogram_1(istft_enhanced_S, fs)
# plot_spectrogram_1(signal, fs)
# plt.plot(signal)
# plt.show()
# plt.plot(istft_enhanced_S)
# plt.show()

gain_py = np.transpose(gain)
wav.write('pink_enhanced_omlsa_imcra_plus.wav', fs, istft_enhanced_S)
