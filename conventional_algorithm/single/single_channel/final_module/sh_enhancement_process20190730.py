import scipy.io.wavfile as wav
from sh_gain_estimation import *
from sh_spectrogram import *
from sh_system_environment import *
from sh_noise_estimation_type import *

# 2019.07.30 이성현 작성

if __name__ == "__main__":
    name = []
    enhance_name = ['Wiener', 'MMSE', 'OMLSA', 'OMLSA_imcra']
    noise_name = ['babble', 'factory', 'white']
    snr_name = ['0dB', '5dB', '10dB']

    n_type = noise_name[0]
    snr_type = snr_name[0]
    EST_TYPE = enhance_name[3]  # 0:Wiener, 1:MMSE_STSA, 2:OM_LSA, 3: noise_PSD_estimation 방법 바꾸기!

    """system start"""
    fs, signal = wav.read('SA1_{}_{}.wav'.format(n_type, snr_type))
    signal = signal/32768

    """system_environment"""
    parameter = {'frm_size':0.032, 'ratio':0.5, 'delta':0.03,'alpha':0.92,'sap':0.5,'k':5}
    win = hann_window(int(fs * parameter['frm_size']))
    stft_Y = stft(signal, fs, win, parameter['frm_size'], parameter['ratio'])

    """noise_estimation"""
    y_PSD = abs(stft_Y) ** 2
    noise_PSD = ImcraTest(y_PSD)
    n_PSD = noise_PSD.noise_estimation_process()
    # n_PSD = average_noise_PSD(stft_Y, parameter['k'])

    # a = n_PSD.transpose()
    # plt.imshow(a)
    # plt.ylim(0,256)
    # plt.show()


    length_waveform = int(fs * parameter['frm_size']) + int((fs * parameter['frm_size'] * parameter['ratio'])*(len(stft_Y) - 1))
    
    """gain_estimation"""

    if EST_TYPE == enhance_name[0]:
        gain = wiener(y_PSD, n_PSD, parameter['alpha'], parameter['delta'])
    elif EST_TYPE == enhance_name[1]:
        gain = mmse_stsa(y_PSD, n_PSD, parameter['delta'], parameter['alpha'], parameter['sap'])
    elif EST_TYPE == enhance_name[2]:
        gain = om_lsa(y_PSD, n_PSD, parameter['delta'], parameter['alpha'])
    elif EST_TYPE == enhance_name[3]:
        gain = om_lsa(y_PSD, n_PSD, parameter['delta'], parameter['alpha'])
    else:
        print("Unsupported option.")
        exit()

    enhanced_S = stft_Y * gain
    istft_enhanced_S = istft(enhanced_S, win, win, length_waveform, parameter['ratio'])*32768
    enhanced_signal = istft_enhanced_S.astype(np.int16)

    wav.write('SA1_{}_{}_enhanced_{}.wav'.format(n_type, snr_type, EST_TYPE), fs, enhanced_signal)

    """print_shape"""
    print(n_PSD.shape)
    print(stft_Y.shape)

"""plot"""
clean_name = 'SA1.wav'
enhanced_name_0 = 'SA1_babble_0dB.wav'
enhanced_name_1 = 'SA1_babble_0dB_enhanced_OMLSA_imcra.wav'
enhanced_name_2 = 'SA1_babble_0dB_enhanced_OMLSA.wav'
enhanced_name_3 = 'SA1_babble_0dB_enhanced_Wiener.wav'

plot_figure = True
fs0, signal0 = wav.read(enhanced_name_0)
fs1, signal1 = wav.read(enhanced_name_1)
fs2, signal2 = wav.read(enhanced_name_2)
fs3, signal3 = wav.read(enhanced_name_3)

if plot_figure:
    fs, clean_signal = wav.read(clean_name)
    plot_spectrogram_4(clean_signal, signal0, signal1, signal2, signal3, fs)


