import numpy as np


def hann_window(winsize_samp):
    """Generate Hann window.

    Args:
        winsize_samp: window length in sample

    Returns:
        window: vector of Hann window
    """
    tmp = np.arange(1, winsize_samp + 1, 1.0, dtype=np.float64)
    window = 0.5 - 0.5 * np.cos((2.0 * np.pi * tmp) / (winsize_samp + 1))
    window = np.float32(window)
    return window

def enframe(waveform, fs, window, frm_size, ratio):  # framing and windowing
    """Framize input waveform.

    Args:
        waveform: input speech signal
        fs: sampling frequency
        window: analysis window
        frm_size: size of frame (sec)
        ratio: overlap ratio (0.25 or 0.5)

    Returns:
        waveform_win: windowed signal
    """

    frm_length_sample = int(frm_size * fs)          # frm_size = 0.016: 16ms , fs = 16000Hz
    inc_sample = int(frm_length_sample * ratio)     # frm_length_sample = 256 , ratio = 0.25 or 0.5

    numfrms = (len(waveform) - frm_length_sample + inc_sample) // inc_sample   # frame number
    waveform_win= np.zeros([numfrms, frm_length_sample])    # (frame_number , frm_length_sample = 256) 만큼 0 넣기
    for frmidx in range(numfrms):
        st = frmidx * inc_sample          # 0, 128, 128*2, 128*3 ...
        waveform_win[frmidx, :] = window * waveform[st:st + frm_length_sample]   # frame index에 따라 window를 256개 단위로 씌워주기

    return waveform_win


def stft(waveform, fs, window_analysis, frm_size, ratio):
    """Perform short-time Fourier transform.

    Args:
        waveform: input signal
        fs: sampling frequency
        window_analysis: analysis window
        frm_size: frame size (sec)
        ratio: overlap ratio (0.25 or 0.5)

    Returns:
        spec_full: complex signal in frequency domain (#frame by FFT size)
    """
    frm = enframe(waveform, fs, window_analysis, frm_size, ratio)
    # spec_full = np.zeros([len(frm),len(frm[0])],dtype=complex) #added

    spec_full = np.fft.fft(frm) #original code - framing and windowing 된 signal 에 fft 해주기
    # for i_frm in range(len(frm)):
    #     spec_full[i_frm] = np.fft.fft(frm[i_frm],2*len(frm[0]))[:int(fs*frm_size)]  #added
    # print(spec_full[:,:257].shape)
    return spec_full  ##[:,:257]

def istft(spec_full, window_analysis, window_synthesis, length_waveform, ratio):
    """Perform inverse short-time Fourier transform.

    Args:
        spec_full: complex signal in frequency domain (#frames by FFT size)
        window_analysis: analysis window
        window_synthesis: synthesis window
        length_waveform: length of synthesized waveform (sample)
        ratio: overlap ratio (0.25 or 0.5)

    Returns:
        waveform: time domain signal
    """
    waveform = np.zeros(length_waveform)
    frm_samp = spec_full.shape[1]
    inc_samp = int(frm_samp * ratio)
    window_mixed = window_analysis * window_synthesis          #     ????????
    window_frag = np.zeros(inc_samp)

    if ratio == 0.5:
        for idx in range(0, 2):
            st = idx * inc_samp
            ed = st + inc_samp
            window_frag += window_mixed[st:ed]
            denorm = np.concatenate([window_frag,
                                     window_frag])
    elif ratio == 0.25:
        for idx in range(0, 4):
            st = idx * inc_samp
            ed = st + inc_samp
            window_frag += window_mixed[st:ed]
            denorm = np.concatenate([window_frag,
                                     window_frag,
                                     window_frag,
                                     window_frag])
    elif ratio == 0.125:
        for idx in range(0, 8):
            st = idx * inc_samp
            ed = st + inc_samp
            window_frag += window_mixed[st:ed]
            denorm = np.concatenate([window_frag,
                                     window_frag,
                                     window_frag,
                                     window_frag,
                                     window_frag,
                                     window_frag,
                                     window_frag,
                                     window_frag])
    else:
        print('only 50%, 75%, 87.5% OLA are available')
    frm = np.real(np.fft.ifft(spec_full))
    for n, i in enumerate(range(0, length_waveform - frm_samp, inc_samp)):
        waveform[i:i + frm_samp] += frm[n] * window_synthesis / denorm
    return waveform