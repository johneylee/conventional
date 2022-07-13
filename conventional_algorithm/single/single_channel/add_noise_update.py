import numpy as np
import random
import scipy
import scipy.io.wavfile as wav
import os
import shutil

def add_noise(sig, noise, snr):
	# choose random section in noise wav file
    ibgn = random.randint(0, len(noise)-len(sig))
    noise = np.array(noise[ibgn:ibgn+len(sig)])

    # scaling factor calculation
    p_sig = np.mean(np.square(sig))
    p_noise = np.mean(np.square(noise))
    scale_factor = np.sqrt((p_sig / np.power(10, snr / 10.0)) / p_noise)

    # add noise to the signal according to the SNR
    noisy = sig + noise * scale_factor
    return noisy




# #original noise adding process
# for i in range(2):
#     # speech, _ = librosa.load('SA'+str(i)+'.wav', sr=sr)
#     # noise, _ = librosa.load('{}.wav'.format(ntype[0]), sr=sr)
#     sampling_rate, speech = wav.read('SA'+str(i)+'.wav')
#     sampling_rate, noise = wav.read('{}.wav'.format(ntype[2]))
#     speech = speech / 32768
#     noise = noise / 32768
#     noisy = add_noise(speech, noise, snr[2])  # SNR: 0,5,10dB
#     scipy.io.wavfile.write('SA'+str(i)+'_{}_{}dB.wav'.format(ntype[2], snr[2]), sr, np.asarray(noisy * 32768, dtype=np.int16))

sr = 16000
snr = [10, 5, 0]
ntype = ["babble", "pink", "factory", "white"]

# txt 리스트 불러와서 한번에 처리하기
file = open("/home/leesunghyun/Downloads/TIMIT/TEST/all_wav.txt", "r")
data = file.read().splitlines()
name = []
for i in range(len(data)):
    sampling_rate, speech = wav.read(data[i])
    sampling_rate, noise = wav.read('{}.wav'.format(ntype[3]))

    speech = speech / 32768.0
    noise = noise / 32768.0
    noisy = add_noise(speech, noise, snr[1])

    name = data[i].split('.')
    filepath = '/home/leesunghyun/Downloads/enhancement/noisy_wavfile/white/5dB'     # change file path
    if not os.path.exists(filepath):            # if not file there
        os.makedirs(filepath)                   # make same path' file

    # os.system('rm -rf %s' % filepath)         # filepath file remove(rm) or filepath folder remove(rm -rf)

    file_name = os.path.basename(os.path.dirname(os.path.dirname(name[0]))) + '_' + os.path.basename(os.path.dirname(name[0])) + '_' + os.path.basename(name[0])
    scipy.io.wavfile.write(filepath + os.sep + file_name + '_{}_{}dB.wav'.format(ntype[3], snr[1]), sr, np.asarray(noisy * 32768, dtype=np.int16))
    # shutil.move("/home/leesunghyun/Downloads/enhancement/noisy_wavfile/")

# #위 작업 이후 생성된 noisy 파일 txt에 리스트로 저장하기
# f = open("/home/leesunghyun/Downloads/enhancement/noisy_wavfile/babble/0dB/noisy_wav.txt", 'w')
# for(path, dir, files) in os.walk("/home/leesunghyun/Downloads/enhancement/noisy_wavfile/babble/0dB"):
#     for filename in files:
#         ext = os.path.splitext(filename)[-1]
#         if ext == '.wav':
#             data = "%s/%s\n" %(path, filename)  # .wav 파일을 한줄씩 all_wav.txt 파일에 저장
#             f.writelines([data])
# f.close()