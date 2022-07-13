import numpy as np
import scipy.io.wavfile as wav
from scipy.special import exp1
from ssp_library import *
import mpmath
import matplotlib.pyplot as plt
from gain_estimation_advanced import *
from spectrogram import *




if __name__ == "__main__":
    fs, signal = wav.read('DR1_FAKS0_SA1_pink_0dB.wav')
    signal = signal/32768
    parameter = {'frm_size':0.032, 'ratio':0.5, 'delta':0.03,'alpha':0.92,'sap':0.5,'k':5}

    win = hann_window(int(fs * parameter['frm_size']))
    stft_Y = stft(signal, fs, win, parameter['frm_size'], parameter['ratio'])
    y_PSD = abs(stft_Y) ** 2

    noise_PSD = ImcraTest(y_PSD)
    n_PSD = noise_PSD.noise_estimation_process()

    a = n_PSD.transpose()
    plt.imshow(a)
    plt.ylim(0,256)
    plt.show()

    print(n_PSD.shape)
    print(stft_Y.shape)
    length_waveform = int(fs * parameter['frm_size']) + int((fs * parameter['frm_size'] * parameter['ratio'])*(len(stft_Y) - 1))

    # gain = wiener(abs(stft_Y) ** 2, n_PSD, parameter['delta'])
    # gain = mmse_stsa(abs(stft_Y) **2, n_PSD, parameter['delta'], parameter['alpha'], parameter['sap'])
    gain = om_lsa(abs(stft_Y)**2, n_PSD, parameter['delta'], parameter['alpha'])
    enhanced_S = stft_Y * gain
    istft_enhanced_S = istft(enhanced_S, win, win, length_waveform, parameter['ratio'])*32768
    istft_enhanced_S = istft_enhanced_S.astype(np.int16)

    plot_spectrogram_1(istft_enhanced_S, fs)
    plot_spectrogram_1(signal, fs)


    # plot_spectrogram_1(signal, fs)
    plt.plot(signal)
    plt.show()
    plt.plot(istft_enhanced_S)
    plt.show()

    gain_py = np.transpose(gain)
    wav.write('pink_enhanced_omlsa_imcra_plus.wav', fs, istft_enhanced_S)
