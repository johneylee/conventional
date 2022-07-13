import numpy as np
import subprocess
import os
import fnmatch
import errno
import mir_eval
from pystoi.stoi import stoi
c = 340


def find_files(directory, pattern):
    matches = []
    for root, dirnames, filenames in os.walk(directory):
        dirnames.sort()
        filenames.sort()
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.join(root, filename.split('.')[0]))
    return matches


def list_to_txt_file(filename, data):
    f = open(filename, 'w')
    for item in data:
        f.write("%s\n" % item)
    f.close()


def read_txt_file(filename):
    f = open(filename, 'r')
    lines = f.read().splitlines()
    f.close()
    return lines


# def get_scale_factor(sig, noise, snr):
#     ibgn = random.randint(0, len(noise)-len(sig))
#     noise = np.array(noise[ibgn:ibgn+len(sig)])
#
#     # scaling factor calculation
#     p_sig = np.mean(np.square(sig))
#     p_noise = np.mean(np.square(noise))
#     scale_factor = np.sqrt((p_sig / np.power(10, snr / 10.0)) / p_noise)
#     return scale_factor


def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def stft(sig, frame_size, overlap_factor=0.5, window=np.hanning, fft_size=512):
    hop_size = int(frame_size*overlap_factor)
    nframe = int(len(sig)/hop_size)
    sig_padded = sig[0:(nframe+1)*hop_size]
    Zxx = np.array([np.fft.fft(window(frame_size) * sig_padded[n:n + frame_size], n=fft_size) for n in range(0, len(sig) - frame_size, hop_size)])
    return Zxx[:, 0:int(fft_size/2)+1]


def istft(Zxx, frame_size, overlap_factor=0.5):
    hopSize = int(frame_size * overlap_factor)
    sig_reconstructed = np.zeros((Zxx.shape[0]+1)*hopSize)
    frm = np.fft.irfft(Zxx)
    for n, i in enumerate(range(0, len(sig_reconstructed) - frame_size, hopSize)):
        sig_reconstructed[i:i + frame_size] += frm[n]
    return sig_reconstructed


def gen_array_signal(signal_stft, mic_array, angle, fs):
    f = np.array([np.linspace(0, fs, 2*(signal_stft.shape[1]-1))[0:signal_stft.shape[1]]])
    steering_vector = np.exp(1j * 2 * np.pi * mic_array * np.cos(angle) * f / c)
    signal_array = np.zeros((signal_stft.shape[0], len(mic_array), signal_stft.shape[1]), dtype='complex_')
    for n in range(signal_stft.shape[0]):
        signal_array[n] = steering_vector * signal_stft[n]
    return signal_array


def gen_gcc_phat(sig0, sig1):
    R = sig0 * np.conj(sig1)
    r_temp = np.fft.irfft(R / np.abs(R))
    r = np.hstack((r_temp[:, int(r_temp.shape[1]/2):], r_temp[:, :int(r_temp.shape[1]/2)]))
    return r


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def objective_measurements(clean, enhanced, fs):
    if len(clean) > len(enhanced):
        clean = clean[:len(enhanced)]
    else:
        enhanced = enhanced[:len(clean)]

    sdr, sir, sar, perm = mir_eval.separation.bss_eval_sources(clean, enhanced)
    score_stoi = stoi(clean, enhanced, fs)
    return sdr[0], sir, sar, perm, score_stoi,


def PESQ(addr_clean, addr_enhanced, fs):
    # exe_dir = '/Users/jinyoung/Storage/1_Study/Code/objective_measurements_for_single_channel_speech_enhancement' \
    #           '/lib/pesq/P862_annex_A_2005_CD/source/'
    exe_dir = '/home/jinyoung/Storage/objective_measurements_for_single_channel_speech_enhancement/lib/' \
              'pesq/P862_annex_A_2005_CD/source/'
    # PESQ = os.system(exe_dir + 'PESQ +' + str(fs) + ' ' + clean + ' ' + enhanced)
    result = subprocess.check_output(exe_dir + 'PESQ +' + str(fs) + ' ' + addr_clean + ' ' + addr_enhanced, shell=True)
    result = result.decode('utf-8')
    score = result.split('MOS-LQO):  = ')[-1]
    Raw_MOS = np.float(score[0:5])
    MOS_LQO = np.float(score[6:-1])
    return Raw_MOS, MOS_LQO
