import numpy as np
from scipy.special import exp1


np.seterr(invalid='ignore')

def average_noise_PSD(y_spec, k):
    """
    :param y_spec: noisy spectrum
    :param k: averaging frame range (1:K)
    :return: average noise PSD
    """
    n_frm, n_freq = y_spec.shape
    N = np.zeros([n_frm, n_freq])
    ave_N = np.mean(abs(y_spec[0:k])**2, 0)

    for i_frm in range(n_frm):
        N[i_frm] = ave_N
    return N


class ImcraTest:
    """
    :param y_PSD: noisy power spectral density
    :return: noise estimate
    :wsub: w frequency smoothing filter
    :u:
    :v:
    :D:120 minimum을 찾기 위한 window length
                -> time:0.032ms * ratio: 하나의 데이터는 16ms 씩 본다
                -> 120 * 16ms = 2초 안에서 최솟값 찾으려면 2초 동안 데이터를 모두 가지고 있어야 한다.
                -> 메모리 너무 많이 사용 / 계산 복잡
                ->  각각의 sub frame 마다 minimum 찾아놓으면 다 저장할 필요없이 그 값만 이용할 수 있어서  u, v 이용
    """
    def __init__(self, y_PSD):
        self.parameter = {'wsub':np.asarray([0.25, 0.5, 0.25]), 'u':8, 'v':15,
                          'beta':1.47, 'Bmin':1.66, 'zeta_0':1.67, 'sapmax':0.95}
        self.alpha = {'0':0.92, 's':0.99, 'd':0.85}
        self.gamma = {'0':4.6, '1':3}

        self.y_PSD = y_PSD
        self.n_frm, self.n_freq = self.y_PSD.shape

        self.n_PSD = np.zeros([self.n_frm, self.n_freq])
        self.s_hat_sub_min = np.zeros([self.parameter['u'], self.n_freq])
        self.s_bar_sub_min = np.zeros([self.parameter['u'], self.n_freq])

        # y_psd 0번째를 noise psd 0번째와 1번째로 초기화
        self.n_PSD[0, :] = self.y_PSD[0, :]
        self.n_PSD[1, :] = self.y_PSD[0, :]

        # frequency domain smoothing initialize
        self.s_f = np.convolve(y_PSD[0, :], self.parameter['wsub'], 'same')
        self.s_hat = self.s_f
        self.s_bar = self.s_f
        self.s_hat_sub = self.s_f
        self.s_bar_sub = self.s_f

        # 120 개로 저장 안하고 8x15로 만들어서 하는데 0이 나오지 않도록 initialize
        for j_frm in range(self.parameter['u']):
            self.s_hat_sub_min[j_frm, :] = self.s_f
            self.s_bar_sub_min[j_frm, :] = self.s_f

        self.s_hat_min = self.s_f
        self.s_bar_min = self.s_f

        # 그 이전 값을 다 speech 로 가정하고 1로 둔 것, distortion 없게 하려고
        self.Gi = np.ones(self.n_freq)                # xi 만드는데 이용되는 gain i
        self.gamma_prev = np.ones(self.n_freq)
        self.n_PSD_bias = self.n_PSD[0, :]            # final function에 이용되는 n_PSD

    """eq14_1"""
    def initialize_smoothing(self, i_frm):
        self.s_f = np.convolve(self.y_PSD[i_frm, :], self.parameter['wsub'], 'same')

        return self.s_f

    """SNR related parameters"""
    def SNR_parameter(self, i_frm):

        # a posteriori SNR = noisy_P / noise_P
        gamma = self.y_PSD[i_frm, :] / self.n_PSD[i_frm, :]
        # Decision-directed, modified version - om lsa
        self.xi = self.alpha['0'] * (self.Gi ** 2) * self.gamma_prev + (1 - self.alpha['0']) * np.maximum(gamma - 1, 0)
        # nu 정의
        self.nu = gamma * self.xi / (1 + self.xi)
        # Gi: spectral gain function - om lsa
        self.Gi = self.xi / (1 + self.xi) * np.exp(exp1(self.nu) / 2)

        return gamma, self.xi, self.nu

    """eq15"""
    def first_smoothing(self, s_f):
        # recursive averaging in time domain smoothing
        self.s_hat = self.alpha['s'] * self.s_hat + (1 - self.alpha['s']) * s_f

        return self.s_hat

    """eq16"""
    def first_minimum(self, ind, s_hat):
        if ind == 1:                                             # v 개의 sample 이 새롭게 시작될때
            self.s_hat_sub = s_hat                               # 이전의 noisy power 사용
        else:
            self.s_hat_sub = np.minimum(s_hat, self.s_hat_sub)   # 그 이후는 min 값 비교하여 power 로 사용
        # print(s_hat.shape)
        # print(s_hat_min.shape)
        self.s_hat_min = np.minimum(s_hat, self.s_hat_min)       # 위 과정을 통해  Smin 값 정해주기 (first smoothing 된 power vs noisy power)
        self.s_hat_sub_min[0, :] = self.s_hat_sub

        return self.s_hat_min

    """eq18,21"""
    def rough_vad(self, i_frm, s_hat, s_hat_min):
        vad_hat = np.zeros(self.n_freq)                          # vad 를 위해 모든 위치에 0
        gamma_hat = self.y_PSD[i_frm, :] / s_hat_min / self.parameter['Bmin']
        zeta_hat = s_hat / s_hat_min / self.parameter['Bmin']
        for i_freq in range(self.n_freq):                        # vad = 1: Speech absence
            if gamma_hat[i_freq] < self.gamma['0'] and zeta_hat[i_freq] < self.parameter['zeta_0']:
                vad_hat[i_freq] = 1

        return vad_hat

    """eq26,27"""
    def second_smoothing(self, vad_hat, i_frm):
        num = np.convolve(self.y_PSD[i_frm, :] * vad_hat, self.parameter['wsub'], 'same')   # vad에 따라 b(i) 만큼의 power convolution
        den = np.convolve(vad_hat, self.parameter['wsub'], 'same')                          # vad값을 b(i) 만큼 convolution

        s_bar_f = num / den                                                                 # smoothing in frequency domain
        for i_freq in range(self.n_freq):
            if den[i_freq] == 0:                                                            # if vad 합이 = 0 이면: speech presence
                s_bar_f = self.s_bar                                                        # 이전값 넣어주기
                                                                                            # otherwise: speech absence
        self.s_bar = self.alpha['s'] * self.s_bar + (1 - self.alpha['s']) * s_bar_f         # smoothing in time domain

        return self.s_bar

    """eq28"""
    def second_minimum(self, ind, s_bar):
        if ind == 1:                                                  # first minimum 과 같은 과정
            self.s_bar_sub = s_bar
        else:
            self.s_bar_sub = np.minimum(s_bar, self.s_bar_sub)

        self.s_bar_min = np.minimum(s_bar, self.s_bar_min)            # first, second smoothing 된 power vs noisy power
        self.s_bar_sub_min[0, :] = self.s_bar_sub

        return self.s_bar_min

    """eq29"""
    def sap(self, i_frm, s_hat, s_bar_min):
        gamma_bar = self.y_PSD[i_frm, :] / s_bar_min / self.parameter['Bmin']
        zeta_bar = s_hat / s_bar_min / self.parameter['Bmin']
        # SAP
        sap = (self.gamma['1'] - gamma_bar) / (self.gamma['1'] - 1)               # if, 1 < gamma_var < gamma_1 & zeta_bar < zeta_0
        for i_freq in range(self.n_freq):
            if zeta_bar[i_freq] >= self.parameter['zeta_0']:                      # otherwise 부분
                sap[i_freq] = 0
        sap = np.minimum(np.maximum(sap, 0), self.parameter['sapmax'])            # sap 의 threshold 설정

        return sap

    """eq29"""
    def spp(self, s_absence_p, xi, nu):
        spp = 1 / (1 + s_absence_p / (1 - s_absence_p) * (1 + xi) * np.exp(-nu))    # 위에서 구한 parameter + sap 이용해서 spp 도출

        return spp


    """eq10,11,12"""
    def noise_PSD_estimation(self, i_frm, s_presence_p):
        alpha_tilde_d = self.alpha['d'] + (1 - self.alpha['d']) * s_presence_p                             # Time-varying smoothing parameter
        self.n_PSD_bias = alpha_tilde_d * self.n_PSD_bias + (1 - alpha_tilde_d) * self.y_PSD[i_frm, :]     # Compute noise power
        # plt.plot(alpha_tilde_d)
        # plt.show()

        # Final function
        self.n_PSD[i_frm + 1, :] = self.parameter['beta'] * self.n_PSD_bias            # Multiplying bias compensation factor

        return self.n_PSD

    """finalization"""
    def finalization(self, ind, gamma):
        if ind == self.parameter['v']:
            # v sample 이 될때마다 u sub window 내에서 minimum 값이 결정되고 저장된다.
            self.s_hat_sub_min[1:self.parameter['u'], :] = self.s_hat_sub_min[:self.parameter['u'] - 1, :]
            self.s_hat_min = np.amin(self.s_hat_sub_min, axis=0)  # 1 -> 0
            self.s_bar_sub_min[1:self.parameter['u'], :] = self.s_bar_sub_min[:self.parameter['u'] - 1, :]
            self.s_bar_min = np.amin(self.s_bar_sub_min, axis=0)

        self.gamma_prev = gamma

    """noise estimation"""
    def noise_estimation_process(self):

        for i_frm in range(1, self.n_frm - 1):  # 첫번째와 마지막 프레임은 제외하고 for 문 돌리겠다.
            ind = i_frm % self.parameter['v'] + 1  # v(15)의 크기만큼이 되면 갱신시켜주기 위한 부분
            s_f = self.initialize_smoothing(i_frm)  # parameter initialize
            gamma, xi, nu = self.SNR_parameter(i_frm)   # define SNR value
            s_hat = self.first_smoothing(s_f)   # frequency & time domain smoothing for vad
            s_hat_min = self.first_minimum(ind, s_hat)
            vad_hat = self.rough_vad(i_frm, s_hat, s_hat_min)
            s_bar = self.second_smoothing(vad_hat, i_frm)
            s_bar_min = self.second_minimum(ind, s_bar)
            s_absence_p = self.sap(i_frm, s_hat, s_bar_min)
            s_presence_p = self.spp(s_absence_p, xi, nu)
            n_PSD = self.noise_PSD_estimation(i_frm, s_presence_p)
            self.finalization(ind, gamma)

        return n_PSD

