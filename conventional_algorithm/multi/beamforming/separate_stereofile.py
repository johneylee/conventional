import os
import sys
import numpy as np
import scipy.io.wavfile as wav
from collections import namedtuple
import torch
from os import makedirs
import scipy

frame_size = 512
overlap_factor = 0.5

def stft(sig, frame_size, overlap_factor=0.5, window=np.hanning, fft_size=512):
    hop_size = int(frame_size*overlap_factor)
    nframe = int(len(sig)/hop_size)
    sig_padded = sig[0:(nframe+1)*hop_size]
    Zxx = np.array([np.fft.fft(window(frame_size) * sig_padded[n:n + frame_size], n=fft_size) for n in range(0, len(sig) - frame_size, hop_size)])
    return Zxx[:, 0:int(fft_size/2)+1]  # spectrogram half


def istft(spec_full, frame_size, overlap_factor=0.5):
    hop_size = int(frame_size * overlap_factor)     # 512 * 0.5 = 256
    reconstructed_wavform = np.zeros((spec_full.shape[0] + 1) * hop_size)  # length wavform: (nframe + 1) * 256
    frm = np.fft.irfft(spec_full)
    for n, i in enumerate(range(0, len(reconstructed_wavform) - frame_size, hop_size)):
        reconstructed_wavform[i:i + frame_size] += frm[n]
    return reconstructed_wavform

for (path, dir, files) in os.walk("/home/leesunghyun/Downloads/enhancement_work/conventional_algorithm/multi_channel/beamforming/data/test"):
    for data in files:
        ext = os.path.splitext(data)[-1]
        if ext == '.wav':
            addr_input = "%s/%s" % (path, data)
            # stft_file = np.load(addr_input)
            # data_out0 = (istft(stft_file[:,0,:], frame_size, overlap_factor) * 32768).astype(np.int16) # write의 target
            #
            # data_out1 = (istft(stft_file[:,1,:], frame_size, overlap_factor) * 32768).astype(np.int16)

            # data_out0 = addr_input[]
            fs, signal = wav.read(addr_input)
            data_out0 = signal[:,0]
            data_out1 = signal[:,1]

            addr_new0 = path.replace('test', 'mic0')
            addr_new1 = path.replace('test', 'mic1')

            makedirs(addr_new0, exist_ok=True)
            makedirs(addr_new1, exist_ok=True)

            # addr_out = addr_input.replace('.wav', '_' + 'e.wav')
            enhanced_out0 = addr_input.replace('test', 'mic0')
            enhanced_out1 = addr_input.replace('test', 'mic1')

            wav.write(enhanced_out0, 16000, data_out0)
            wav.write(enhanced_out1, 16000, data_out1)