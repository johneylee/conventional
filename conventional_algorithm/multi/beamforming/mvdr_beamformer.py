import numpy as np
import os
import scipy.io.wavfile as wav
from tqdm import tqdm
import sys
sys.path.insert(0, "../")
import ssplib
from os import makedirs

def istft(spec_full, frame_size, overlap_factor=0.5):
    hop_size = int(frame_size * overlap_factor)     # 512 * 0.5 = 256
    reconstructed_wavform = np.zeros((spec_full.shape[0] + 1) * hop_size)  # length wavform: (nframe + 1) * 256
    frm = np.fft.irfft(spec_full)
    for n, i in enumerate(range(0, len(reconstructed_wavform) - frame_size, hop_size)):
        reconstructed_wavform[i:i + frame_size] += frm[n]
    return reconstructed_wavform

def get_beam_pattern(w, f):
    psi = np.linspace(0, np.pi, 181).reshape([1, -1])
    bp_steering_vector = np.exp(1j * 2 * np.pi * mic_array * np.cos(psi) * f / c)
    beam_pattern = 1/w.shape[0] * w.conj().T @ bp_steering_vector
    beam_pattern = np.reshape(beam_pattern, [-1])
    return beam_pattern


def dsp_beamformer(w, mch_stft):
    enhanced_stft = np.mean(w.conj() * mch_stft, 1)
    return w, enhanced_stft


def mvdr_beamformer_fl(h, mch_stft):
    # Ideally, have to use Rnn, but under some conditions,
    # using Rxx is same (minimum power distortionless response, MPDR)

    # frame level version
    w = np.zeros((mch_stft.shape[0], mch_stft.shape[1], mch_stft.shape[2]), dtype="complex_")
    for n in range(mch_stft.shape[0]):
        Rxx = mch_stft[n] @ mch_stft[n].conj().T + np.eye(mch_stft.shape[1])
        Rxx = (Rxx + Rxx.T) / 2  # ensure Hermitian
        Rxx_inverse = np.linalg.inv(Rxx)
        for m in range(mch_stft.shape[2]):
            h_m = np.array([h[:, m]]).T
            w[n, :, m] = np.linalg.inv(h_m.conj().T @ Rxx_inverse @ h_m) * (Rxx_inverse @ h_m)[0]

    enhanced_stft = np.sum(w.conj() * mch_stft, 1)
    return w, enhanced_stft


def mvdr_beamformer_sl(h, mch_stft):  # This is the one. It is same whether np.eye exists or not
    # Ideally, have to use Rnn, but under some conditions,
    # using Rxx is same (minimum power distortionless response, MPDR)

    # sentence level version
    w = np.zeros((mch_stft.shape[1], mch_stft.shape[2]), dtype="complex_")
    Rxx = np.zeros((mch_stft.shape[0], mch_stft.shape[1], mch_stft.shape[1]), dtype="complex_")
    for n in range(mch_stft.shape[0]):
        Rxx[n] = mch_stft[n] @ mch_stft[n].conj().T + np.eye(mch_stft.shape[1])
    Rxx = np.mean(Rxx, 0)
    Rxx = (Rxx + Rxx.T) / 2  # ensure Hermitian
    Rxx_inverse = np.linalg.inv(Rxx)
    for m in range(mch_stft.shape[2]):
        h_m = np.array([h[:, m]]).T
        w[:, m] = np.linalg.inv(h_m.conj().T @ Rxx_inverse @ h_m) * (Rxx_inverse @ h_m)[0]

    enhanced_stft = np.sum(w.conj() * mch_stft, 1)
    return w, enhanced_stft


def gsc_beamformer(steering_vector, array_signal, wc):
    h = steering_vector
    x1 = array_signal
    n = array_signal.shape[0]
    A = np.zeros((n-1, 1))
    mu = 0.005

    # 1. conventional beamforming
    yc = wc.conj().T @ x1

    # 2. blocking
    B = np.hstack((np.eye(n-1), np.zeros((n-1, 1)))) + np.hstack((np.zeros((n-1, 1)), -1*np.eye(n-1)))
    x2 = B @ x1

    # 3. LMS
    for n in range(100):
        ya = A @ x2
        y = yc - ya
        A = A + mu*y*x2
    return


# model_name0 = 'delay_and_sum'
# model_name1 = 'IRM_MVDR_sl_eye'
# model_name2 = 'MVDR_fl_eye'
# model_dir0 = os.path.join('Statistical', model_name0)
# model_dir1 = os.path.join('Statistical', model_name1)
# model_dir2 = os.path.join('Statistical', model_name2)
# test_name = 'TEST_CORE_directive_2mic'
# result_dir0 = os.path.join(model_dir0, test_name)
# result_dir1 = os.path.join(model_dir1, test_name)
# result_dir2 = os.path.join(model_dir2, test_name)
# project_dir = os.getcwd()







fs = 16000
frame_time = 32
frame_size = int(fs*frame_time/1000)
frequency_bin = np.linspace(0, fs, frame_size)
frequency_bin = frequency_bin[:int(frame_size/2+1)]

c = 340
mic_num = 2
mic_dist = 0.08
mic_array = np.linspace(0, (mic_num-1)*mic_dist, mic_num).reshape([-1, 1])
theta = np.pi / 2
steering_vector = np.exp(1j * 2 * np.pi * mic_array * np.cos(theta) * frequency_bin / c)

# list_name = 'test_noisy_list.txt'
# list_name = '../DB/enhanced_directive_noisymchstft/enhanced_wav_list.txt'
# test_list = ssplib.read_txt_file(list_name)
# test_num = len(test_list)
# enhanced_wav_list = []
# enhanced_wav_list1 = []
# enhanced_wav_list2 = []
#
# for n in tqdm(range(len(test_list))):
#     mch_stft = np.load(test_list[n] + '.noisyMchSTFT.npy')
#     w_MVDR, enhanced_stft_MVDR = mvdr_beamformer_sl(steering_vector, mch_stft)
#     enhanced_wav_MVDR = ssplib.istft(enhanced_stft_MVDR, frame_size, 0.5)
#     name = test_list[n].split('/')
#     name2 = os.path.join(result_dir1, name[-3], name[-2], name[-1]) + '_e.wav'
#     ssplib.make_sure_path_exists(os.path.dirname(name2))
#     wav.write(name2, fs, np.asarray(enhanced_wav_MVDR * 32768, dtype=np.int16))
#     enhanced_wav_list1.append(os.path.join(project_dir, name2))
#
# ssplib.list_to_txt_file(os.path.join(result_dir1, 'enhanced_wav_list.txt'), enhanced_wav_list1)

# full name
for (path, dir, files) in os.walk("/home/leesunghyun/Downloads/enhancement_work/conventional_algorithm/multi_channel/beamforming/data_joint/SvsM/stftnumpy"):
    for data in files:
        ext = os.path.splitext(data)[-1]
        if ext == '.npy':
            addr_input = "%s/%s" % (path, data)
            # data load
            mch_stft = np.load(addr_input)
            w_MVDR, enhanced_stft_MVDR = mvdr_beamformer_sl(steering_vector, mch_stft)
            enhanced_wav_MVDR = istft(enhanced_stft_MVDR, frame_size, 0.5).astype(np.int16)
            # making new folder and file name
            save_result_folder = path.replace('stftnumpy', 'single2mvdr_notRIR')
            save_result_folder_change = save_result_folder.replace('-', 'm')
            makedirs(save_result_folder_change, exist_ok=True)
            data_split = data.split('_')
            new_data_name = data_split[0] + '_' + data_split[1] + '_' + data_split[2] + '.wav'
            addr_out = "%s/%s" % (save_result_folder_change, new_data_name)
            # addr_out = name_change.replace('_joint2.noisyMchSTFT.npy', '.wav')
            wav.write(addr_out, fs, enhanced_wav_MVDR)

# # short name
# for (path, dir, files) in os.walk("/home/leesunghyun/Downloads/enhancement_work/conventional_algorithm/multi_channel/beamforming/stftnumpy"):
#     for data in files:
#         ext = os.path.splitext(data)[-1]
#         if ext == '.npy':
#             addr_input = "%s/%s" % (path, data)
#             # data load
#             mch_stft = np.load(addr_input)
#             w_MVDR, enhanced_stft_MVDR = mvdr_beamformer_sl(steering_vector, mch_stft)
#             enhanced_wav_MVDR = istft(enhanced_stft_MVDR, frame_size, 0.5).astype(np.int16)
#             # making new folder and file name
#             save_result_folder = path.replace('stftnumpy', 'single2mvdr_2')
#             makedirs(save_result_folder, exist_ok=True)
#             box = data.split('_')
#             data_name_change = box[0] + '_' + box[1] + '_' + box[2] + '.wav'
#             addr_out = "%s/%s" % (save_result_folder, data_name_change)
#             wav.write(addr_out, fs, enhanced_wav_MVDR)
