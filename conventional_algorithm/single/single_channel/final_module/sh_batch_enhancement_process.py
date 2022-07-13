import scipy.io.wavfile as wav
from sh_gain_estimation import *
from sh_system_environment import *
from sh_noise_estimation_type import *
import os
from glob import glob
import scipy

'''
    Implementation of Speech Enhancement System(Batch)

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
    # noise_name = ['babble', 'factory', 'pink', 'white']
    noise_name = ['destroyerengine', 'destroyerops', 'factory2', 'leopard', 'm109', 'machinegun']
    noise_esti_name = ['Original', 'IMCRA']
    # snr_name = ['0dB', '5dB', '10dB']
    snr_name = ['m5dB', '0dB', '5dB', '10dB']

    n_type = noise_name[0]
    snr_type = snr_name[0]
    n_e_type = noise_esti_name[1] # 0: Original, 1: IMCRA
    EST_TYPE = enhance_name[2]  # 0:Wiener, 1:MMSE_STSA, 2:OM_LSA

    # infile_name = glob("/home/leesunghyun/Downloads/enhancement/noisy_wavfile/" + n_type + os.sep + snr_type + os.sep + "*.wav")
    infile_name = glob(
        "/home/leesunghyun/Downloads/IRM/workspace/data/speech/test/noisy/" + n_type + os.sep + snr_type + os.sep + "*.wav")

    for i in range(len(infile_name)):
        fs, signal = wav.read(infile_name[i])
        name = infile_name[i].split('.')

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

        """File saving process"""
        if EST_TYPE == enhance_name[2]:
            filepath = '/home/leesunghyun/Downloads/enhancement/enhanced_wavfile2/' + EST_TYPE + os.sep + n_type + os.sep + snr_type + os.sep + n_e_type
            print(filepath)
            file_name = os.path.basename(name[0])
            scipy.io.wavfile.write(filepath + os.sep + file_name + '_{}_{}.wav'.format(EST_TYPE, n_e_type), fs, np.asarray(enhanced_signal, dtype=np.int16))
        else:
            filepath = '/home/leesunghyun/Downloads/enhancement/enhanced_wavfile2/' + EST_TYPE + os.sep + n_type + os.sep + snr_type
            print(filepath)
            file_name = os.path.basename(name[0])
            scipy.io.wavfile.write(filepath + os.sep + file_name + '_{}.wav'.format(EST_TYPE), fs, np.asarray(enhanced_signal, dtype=np.int16))
