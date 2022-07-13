# from ssplib import *
import os
from glob import glob
import scipy.io.wavfile as wav
import scipy
from gain_estimate import *
from spectrogram import *

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
    # wsub = np.asarray([[0.25],[0.5],[0.25]])
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
    gamma_2 = gamma_1 - 1    # ??

    sapmax = 0.95            # ??

    s_hat_sub_min = np.zeros([u, n_freq])
    s_bar_sub_min = np.zeros([u, n_freq])

    ## initialization
    n_PSD[0, :] = y_PSD[0, :]                         ################
    n_PSD[1, :] = y_PSD[0, :]                         ################
    s_f = np.convolve(y_PSD[0, :], wsub, 'same')
    s_hat = s_f
    s_bar = s_f
    s_hat_sub = s_f
    s_bar_sub = s_f

    for j_frm in range(u):
        s_hat_sub_min[j_frm, :] = s_f
        s_bar_sub_min[j_frm, :] = s_f

    s_hat_min = s_f
    s_bar_min = s_f

    Gi = np.ones(n_freq)                              ################
    gamma_prev = np.ones(n_freq)
    n_PSD_bias = n_PSD[0, :]                            ####################3

    ## noise estimation
    for i_frm in range(1, n_frm - 1):                 ###################
        ind = i_frm % v + 1
        s_f = np.convolve(y_PSD[i_frm, :], wsub, 'same')

        ## SNR related params
        gamma      = y_PSD[i_frm, :] / n_PSD[i_frm, :]
        xi         = alpha * (Gi ** 2) * gamma_prev + (1 - alpha) * np.maximum(gamma-1, 0)
        nu         = gamma * xi / (1 + xi)
        Gi = xi / (1 + xi) * np.exp(exp1(nu) / 2)

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
        s_hat_sub_min[0, :] = s_hat_sub                     ######################

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
        s_bar_sub_min[0, :] = s_bar_sub                         #######################3

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
    hlocal = hlocal / sum(hlocal)
    hglobal = hann_window(2 * wglobal + 1)
    hglobal = hglobal / sum(hglobal)

    zetamin = 10 ** (-10 / 10)
    zetamax = 10 ** (-5 / 10)
    zetaratio = np.log(zetamax / zetamin)

    zetapmin = 10 ** (0 / 10)
    zetapmax = 10 ** (10 / 10)

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
        gain[i_frm, :] = np.minimum((gain_i_frm ** spp) * (delta ** (1 - spp)) ,1)

        ##finalization
        xi_old = (gain_i_frm ** 2) * gamma
        xi_prev = xi
        zetaframeprev = zetaframe
        zetaprev = zeta

    return gain


if __name__ == '__main__':
    name = []
    enhance_name = 'OMLSA_IMCRA3'
    noise_name = 'factory'
    snr_type = '0dB'
    infile_name = glob("/home/leesunghyun/Downloads/enhancement/noisy_wavfile/" + noise_name + os.sep + snr_type + os.sep + "*.wav")
    for i in range(len(infile_name)):
        fs, signal = wav.read(infile_name[i])
        name = infile_name[i].split('.')

        signal = signal / 32768
        frm_size = 0.032
        ratio = 0.5
        delta = 0.03
        alpha = 0.92
        sap = 0.5
        k = 5
        win = hann_window(int(fs * frm_size))
        stft_Y = stft(signal, fs, win, frm_size, ratio)
        n_PSD = imcra(abs(stft_Y) ** 2)
        # n_PSD = average_noise_PSD(stft_Y, k)

        print(n_PSD.shape)
        print(stft_Y.shape)
        length_waveform = int(fs * frm_size) + int((fs * frm_size * ratio) * (len(stft_Y) - 1))

        # gain = wiener(abs(stft_Y) ** 2, n_PSD, delta)
        # gain = mmse_stsa(abs(stft_Y) ** 2, n_PSD, delta, alpha, sap)
        gain = om_lsa(abs(stft_Y) ** 2, n_PSD, delta, alpha)
        enhanced_S = stft_Y * gain

        istft_enhanced_S = istft(enhanced_S, win, win, length_waveform, ratio) * 32768
        enhanced_signal = istft_enhanced_S.astype(np.int16)
        gain_py = np.transpose(gain)

        filepath = '/home/leesunghyun/Downloads/enhancement/enhanced_wavfile/' + enhance_name + os.sep + noise_name + os.sep + snr_type
        file_name = os.path.basename(name[0])
        scipy.io.wavfile.write(filepath + os.sep + file_name + '_OMLSA_IMCRA3.wav', fs, np.asarray(enhanced_signal, dtype=np.int16))