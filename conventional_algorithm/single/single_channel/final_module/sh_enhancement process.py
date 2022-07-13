import scipy.io.wavfile as wav
from scipy.signal import wiener
from sh_gain_estimation import *
from sh_spectrogram import *
from sh_system_environment import *
from sh_noise_estimation_type import *

'''
    Implementation of Speech Enhancement System(Small)
    
    Contents.
        - Wiener gain   
        - MMSE
        - OMLSA
            Original
            IMCRA  
    Authors.
        Sung-Hyun Lee, DSP&Ai Lab., Yonsei University
        Hong-Goo Kang, DSP&AI Lab., Yonsei University
        
    Date. July 31st, 2019.
'''



if __name__ == "__main__":
    name = []
    enhance_name = ['Wiener', 'MMSE', 'OMLSA']
    noise_esti_name = ['Original', 'IMCRA']
    noise_name = ['babble', 'factory', 'pink']
    snr_name = ['0dB', '5dB', '10dB']

    n_type = noise_name[0]
    snr_type = snr_name[0]
    n_e_type = noise_esti_name[1] # 0: Original, 1: IMCRA
    EST_TYPE = enhance_name[0]  # 0:Wiener, 1:MMSE_STSA, 2:OM_LSA

    """system start"""
    fs, signal = wav.read('SA1_{}_{}.wav'.format(n_type, snr_type))
    signal = signal / 32768

    """system_environment"""
    parameter = {'frm_size': 0.032, 'ratio': 0.5, 'k': 5}
    win = hann_window(int(fs * parameter['frm_size']))
    stft_Y = stft(signal, fs, win, parameter['frm_size'], parameter['ratio'])

    """noise_estimation"""
    y_PSD = abs(stft_Y) ** 2

    if n_e_type == noise_esti_name[0]:
        noise_PSD = ImcraTest(y_PSD)
        n_PSD = noise_PSD.noise_estimation_process()
    elif n_e_type == noise_esti_name[1]:
        n_PSD = average_noise_PSD(stft_Y, parameter['k'])

    # a = n_PSD.transpose()
    # plt.imshow(a)
    # plt.ylim(0,256)
    # plt.show()

    length_waveform = int(fs * parameter['frm_size']) + int((fs * parameter['frm_size'] * parameter['ratio']) * (len(stft_Y) - 1))

    """gain_estimation"""

    if EST_TYPE == enhance_name[0]:
        method1 = gain_estimation(y_PSD, n_PSD)
        gain = method1.wiener()

    elif EST_TYPE == enhance_name[1]:
        method2 = gain_estimation(y_PSD, n_PSD)
        gain = method2.mmse_stsa()

    elif EST_TYPE == enhance_name[2]:
        method3 = gain_estimation(y_PSD, n_PSD)
        gain = method3.om_lsa()

    else:
        print("Unsupported option.")
        exit()

    enhanced_S = stft_Y * gain
    istft_enhanced_S = istft(enhanced_S, win, win, length_waveform, parameter['ratio']) * 32768
    enhanced_signal = istft_enhanced_S.astype(np.int16)

    wav.write('SA1_{}_{}_enhanced_{}_{}.wav'.format(n_type, snr_type, EST_TYPE, n_e_type), fs, enhanced_signal)

    """print_shape"""
    print(n_PSD.shape)
    print(stft_Y.shape)
    print('SA1_{}_{}_enhanced_{}_{}.wav'.format(n_type, snr_type, EST_TYPE, n_e_type))


"""plot"""
clean_name = 'SA1.wav'
enhanced_name_0 = 'SA1_babble_0dB.wav'
enhanced_name_1 = 'SA1_babble_0dB_enhanced_OMLSA_IMCRA.wav'
enhanced_name_2 = 'SA1_babble_0dB_enhanced_OMLSA_Original.wav'
enhanced_name_3 = 'SA1_babble_0dB_enhanced_MMSE_Original.wav'
# enhanced_name_3 = 'SA1_pink_0dB_enhanced_Wiener.wav'

plot_figure = True
fs0, signal0 = wav.read(enhanced_name_0)
fs1, signal1 = wav.read(enhanced_name_1)
fs2, signal2 = wav.read(enhanced_name_2)
fs3, signal3 = wav.read(enhanced_name_3)

if plot_figure:
    fs, clean_signal = wav.read(clean_name)
    plot_spectrogram_4(clean_signal, signal0, signal1, signal2, signal3, fs)


