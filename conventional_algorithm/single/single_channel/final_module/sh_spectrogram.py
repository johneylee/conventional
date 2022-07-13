import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

def plot_spectrogram_1(input_signal, fs):
    frm_size = 0.032
    # ratio = 0.9
    ratio = 0.5
    n_fft = int(fs * frm_size)
    n_overlap = int(fs * frm_size * ratio)
    win = np.hanning(n_fft)

    plt.figure()
    plt.specgram(input_signal, NFFT=n_fft, Fs=int(fs), window=win, noverlap=n_overlap, cmap='jet', vmin=-20, vmax=40)
    plt.colorbar()
    plt.title('Input signal')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.show()

def plot_spectrogram_4(clean_signal, input_signal, enhanced_1, enhanced_2, enhanced_3, fs):
    frm_size = 0.032
    ratio = 0.9
    n_fft = int(fs * frm_size)
    n_overlap = int(fs * frm_size * ratio)
    win = np.hanning(n_fft)

    plt.figure(2)
    # plt.subplot(311)
    # plt.specgram(clean_signal, NFFT=n_fft, Fs=int(fs), window=win, noverlap=n_overlap, cmap='jet', vmin=-30, vmax=60)
    plt.specgram(clean_signal, NFFT=n_fft, Fs=int(fs), window=win, noverlap=n_overlap, cmap='jet', vmin=-20, vmax=40)
    plt.colorbar()
    plt.title('Clean signal')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.show()

    # plt.subplot(312)
    plt.specgram(input_signal, NFFT=n_fft, Fs = int(fs), window=win, noverlap=n_overlap, cmap='jet', vmin=-20, vmax=40)
    plt.colorbar()
    plt.title('Clean')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.show()

    # plt.subplot(313)
    plt.specgram(enhanced_1, NFFT=n_fft, Fs=int(fs), window=win, noverlap=n_overlap, cmap='jet', vmin=-20, vmax=40)
    plt.colorbar()
    plt.title('Noisy')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.show()

    plt.specgram(enhanced_2, NFFT=n_fft, Fs=int(fs), window=win, noverlap=n_overlap, cmap='jet', vmin=-20, vmax=40)
    plt.colorbar()
    plt.title('IBM_epoch50')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.show()

    plt.specgram(enhanced_3, NFFT=n_fft, Fs=int(fs), window=win, noverlap=n_overlap, cmap='jet', vmin=-20, vmax=40)
    plt.colorbar()
    plt.title('IBM_epoch100')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.show()


"""plot"""


clean_name = 'SA1.wav'
# enhanced_name_0 = 'SA1_babble_0dB.wav'
# enhanced_name_1 = 'SA1_babble_0dB_enhanced_OMLSA_IMCRA.wav'
enhanced_name_0 = 'FDHC0_SX389.wav'
enhanced_name_1 = 'FDHC0_SX389_noisy.wav'
# enhanced_name_2 = 'SA1_babble_0dB_enhanced_OMLSA_Original.wav'
# enhanced_name_3 = 'SA1_babble_0dB_enhanced_MMSE_Original.wav'
# enhanced_name_3 = 'SA1_pink_0dB_enhanced_Wiener.wav'

# test
enhanced_name_2 = 'FDHC0_SX389_sample2_ibm_ibm.wav'
enhanced_name_3 = 'FDHC0_SX389_sample3_ibm_ibm.wav'


plot_figure = True
fs0, signal0 = wav.read(enhanced_name_0)
fs1, signal1 = wav.read(enhanced_name_1)
fs2, signal2 = wav.read(enhanced_name_2)
fs3, signal3 = wav.read(enhanced_name_3)

if plot_figure:
    fs, clean_signal = wav.read(clean_name)
    plot_spectrogram_4(clean_signal, signal0, signal1, signal2, signal3, fs)

# if __name__ == '__main__':
#     infile_name = 'DR1_FAKS0_SI943_factory_0dB.wav'
#     clean_name = 'SI943.wav'
#     direct_signal = 'DR1_FAKS0_SI943_factory_0dB_OMLSA.wav'
#     irm_signal = 'DR1_FAKS0_SI943_factory_0dB_OMLSA_IMCRA2.wav'
#     ibm_signal = 'DR1_FAKS0_SI943_factory_0dB_OMLSA_IMCRA3.wav'
#
#
#     play_audio = True   # Play input/output audio files if True.
#     plot_figure = True   # Plot figures if True.
#
#     fs, signal = wav.read(infile_name)
#     fs2, signal2 = wav.read(direct_signal)
#     fs3, signal3 = wav.read(irm_signal)
#     fs4, signal4 = wav.read(ibm_signal)
#
#     if plot_figure:
#         fs, clean_signal = wav.read(clean_name)
#         plot_spectrogram(clean_signal, signal, signal2, signal3, signal4, fs)