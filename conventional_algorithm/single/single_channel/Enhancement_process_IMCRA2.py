# Required libraries
#   - scipy: to handle wave file format
#   - numpy: to handle array structure
#   - sounddevice: to play sound
import scipy.io.wavfile as wav
import numpy as np
import sounddevice as sd
from enh_lib import *
import matplotlib.pyplot as plt
import scipy
import scipy.io.wavfile as wav
import os
from glob import glob
from ssp_library import *
from gain_estimation_advanced import *

# from add_noise import *

# Define input and output file names.

# clean_name = 'SA1.wav'

# Set flags for debugging.
play_audio = True   # Play input/output audio files if True.
plot_figure = True   # Plot figures if True.


# Read a wav input file and open a file handle to write outputs.
# file = open("/home/leesunghyun/Downloads/enhancement/noisy_wavfile/factory/0dB/noisy_wav.txt", "r")
# infile_name = file.read().splitlines()
name = []

enhance_name = 'LSA'
noise_name = 'pink'
snr_type = '10dB'
# infile_name = glob("/home/leesunghyun/Downloads/enhancement/noisy_wavfile/factory/0dB/*.wav")
infile_name = glob("/home/leesunghyun/Downloads/enhancement/noisy_wavfile/" + noise_name + os.sep + snr_type + os.sep + "*.wav")
for i in range(len(infile_name)):
    fs, signal = wav.read(infile_name[i])
    # .wav and path split
    name = infile_name[i].split('.')

    # Set environmental variables.
    frame_size = 0.032  # 32ms (unit = second)
    overlap_ratio = 0.5  # ratio (range between 0.5 and 1)
    frame_len = int(fs * frame_size)  # frame length: 16kHz * 0.032s = 512
    overlap_len = int(frame_len * overlap_ratio)  # overlap length: 512 * 0.5 = 256
    shift_len = frame_len - overlap_len  # shift length
    norm_factor = frame_len / (2 * shift_len)  # normalization factor for overlap-and-add processing
    num_frames = int((len(signal) - frame_len) / shift_len)  # total number of frames
    fft_len = frame_len  # Need to modify if frame_len is not power of two
    fft_len_half = fft_len // 2 + 1  # half of the FFT size

    # Set gain estimator related variables.
    EST_TYPE = 7  # 0: spectral subtraction, 1: power subtraction, 2: Wiener, 3: MMSE-STSA, 4: ML, 5: MLSD, 6:MLSD TEST 7:MMSE-LSA
    NOISE_PSD_FRAME_NO = 5  # number of first N frames to compute noise PSD
    MIN_GAIN = 1e-3  # minimum gain value
    q = 1 / 2
    snr = 0

    # Prepare (Hanning) window function, noise PSD buffer, and output signal buffer.
    window = np.hanning(frame_len)  # hann window
    noise_psd = np.zeros(fft_len_half)  # noise PSD buffer
    synthesis_waveform = np.zeros(len(signal))  # synthesis waveform buffer
    prev_x_hat = np.zeros(frame_len)  # buffer for overlap and add

    # Prepare memory for enhancement processing.
    enhanced_hat = np.zeros(fft_len)
    phs_hat = np.zeros(fft_len)
    if EST_TYPE == 1 or EST_TYPE == 2 or EST_TYPE == 3 or EST_TYPE == 4 or EST_TYPE == 5 or EST_TYPE == 6 or EST_TYPE == 7:
        ALPHA = 0.9
        SAP = 0.5
        prev_xi = np.ones(fft_len_half)

    # Perform a frame-by-frame processing.
    for i_frame in range(num_frames):
        iBgn = i_frame * shift_len  # beginning index of the window
        iEnd = iBgn + frame_len  # ending index of the window
        win_waveform = window * signal[iBgn:iEnd]  # windowed signal

        # Perform fast Fourier transform (FFT).
        # stft_Y = np.fft.fft(win_waveform) # 원래 코드
        stft_Y = stft(signal, fs, win, frm_size, ratio)

        # Convert into the polar coordinate (need only half of the frequency bins).
        mag_spectrum = np.abs(stft_Y[:fft_len_half])
        phs_spectrum = np.angle(stft_Y[:fft_len_half])
        pow_spectrum = mag_spectrum ** 2

        # Compute noise PSD using the first N frames.
        # if i_frame < NOISE_PSD_FRAME_NO:
        #     noise_psd = estimate_noise_psd(noise_psd, pow_spectrum, i_frame) # 이부분에서 추정된 noise psd 넣어주므로 수정 필요
        #                                                                      # but, frame number 5개가 아닌 전체를 따져줘야하므로 if 문 내부인지 다시 생각
        # Using IMCRA
        noise_psd = imcra(abs(stft_Y) ** 2)

        # Estimate gain values and obtain enhanced magnitude spectrum.
        if EST_TYPE == 0:  # spectral subtraction
            gain = spectral_subtraction(pow_spectrum, noise_psd, MIN_GAIN)
        elif EST_TYPE == 1:  # Power subtraction
            prev_xi, gain = Power_sub(pow_spectrum, noise_psd, prev_xi, ALPHA, MIN_GAIN)
        elif EST_TYPE == 2:  # Wiener
            prev_xi, gain = wiener(pow_spectrum, noise_psd, prev_xi, ALPHA, MIN_GAIN)
        elif EST_TYPE == 3:  # MMSE-STSA
            prev_xi, gain = mmse_stsa(pow_spectrum, noise_psd, prev_xi, ALPHA, SAP, MIN_GAIN)
        elif EST_TYPE == 4:  # ML
            gain = ml(pow_spectrum, noise_psd, MIN_GAIN)
        elif EST_TYPE == 5:  # ML_SD - prev
            gain = ml_sd(pow_spectrum, noise_psd, MIN_GAIN, snr, q)
        elif EST_TYPE == 6:  # ML_SD - last : test
            gain = ml_soft_decision(pow_spectrum, noise_psd, MIN_GAIN)
        elif EST_TYPE == 7:  # MMSE-LSA
            prev_xi, gain = mmse_lsa(pow_spectrum, noise_psd, prev_xi, ALPHA, MIN_GAIN)
        else:
            print("Unsupported option.")
            exit()

        enhanced_spectrum = mag_spectrum * gain

        # Generate magnitude and phase spectrum to process inverse FFT.
        enhanced_hat[:fft_len_half] = enhanced_spectrum
        enhanced_hat[fft_len_half:] = enhanced_hat[fft_len_half - 2:0:-1]
        phs_hat[:fft_len_half] = phs_spectrum
        phs_hat[fft_len_half:] = -phs_hat[fft_len_half - 2:0:-1]

        # Convert back to the rectangular coordinate.
        enhanced_Y = enhanced_hat * np.exp(1j * phs_hat)

        # Perform inverse FFT.
        reconstructed_waveform = np.real(np.fft.ifft(enhanced_Y))

        # Perform overlap-and-add processing and adjust buffer for next frame processing.
        segment = prev_x_hat + reconstructed_waveform
        prev_x_hat[:overlap_len] = segment[shift_len:]

        # Save the reconstructed signal into the output buffer.
        synthesis_waveform[iBgn:iBgn + shift_len] = segment[:shift_len] / norm_factor

    # Write the remained data (in the buffer) into the output file.
    # May not be needed in real application systems.
    iBgn = num_frames * shift_len
    synthesis_waveform[iBgn:iBgn + overlap_len] = prev_x_hat[:overlap_len] / norm_factor

    # Write synthesized signal into a wav file (binary 16 bits/sample).
    enhanced_signal = synthesis_waveform.astype(np.int16)

    # change path
    # filepath = '/home/leesunghyun/Downloads/enhancement/enhanced_wavfile/Wiener/factory/0dB'
    filepath = '/home/leesunghyun/Downloads/enhancement/enhanced_wavfile/' + enhance_name + os.sep + noise_name + os.sep + snr_type

    if not os.path.exists(filepath):  # if not file there
        os.makedirs(filepath)  # make same path' file
    # naming for enhanced signal
    file_name = os.path.basename(name[0])
    # file_name = os.path.basename(os.path.dirname(os.path.dirname(name[0]))) + '_' + os.path.basename(os.path.dirname(name[0])) + '_' + os.path.basename(name[0])
    # make .wav file for enhanced signal
    scipy.io.wavfile.write(filepath + os.sep + file_name + '_LSA.wav', fs, np.asarray(enhanced_signal, dtype=np.int16))


# # 위 작업 이후 생성된 noisy 파일 txt에 리스트로 저장하기
# f = open("/home/leesunghyun/Downloads/enhancement/enhanced_wavfile/Wiener/factory/0dB/enhanced_wav.txt", 'w')
# for(path, dir, files) in os.walk("/home/leesunghyun/Downloads/enhancement/enhanced_wavfile/Wiener/factory/0dB"):
#     for filename in files:
#         ext = os.path.splitext(filename)[-1]
#         if ext == '.wav':
#             data = "%s/%s\n" %(path, filename)  # .wav 파일을 한줄씩 all_wav.txt 파일에 저장
#             f.writelines([data])
# f.close()
