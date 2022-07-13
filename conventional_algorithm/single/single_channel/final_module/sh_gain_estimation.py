from scipy.special import i0, i1 # modified bessel function
from scipy.special import exp1
from sh_system_environment import *
import numpy as np

np.seterr(invalid='ignore')

class gain_estimation:
    def __init__(self, y_PSD, n_PSD):
        self.parameter = {'delta':0.03,'alpha':0.92,'sap':0.5}
        self.y_PSD = y_PSD
        self.n_PSD = n_PSD

    def spectral_subtraction(self):
        """
        :param y_PSD: noisy power spectral density
        :param n_PSD: noise power spectral density
        :param delta: minimum gain threshold
        :return: Spectral Subtraction gain
        """
        return np.maximum(1 - np.sqrt(self.n_PSD / self.y_PSD), self.parameter['delta'])

    def wiener(self):
        """
        :param y_PSD: noisy power spectral density
        :param n_PSD: noise power spectral density
        :param alpha: alpha in DD approach
        :param delta: minimum gain threshold
        :return: Wiener gain
        """
        n_frm, n_freq = self.y_PSD.shape
        prev_xi = 1
        xi = np.zeros([n_frm, n_freq])
        for i_frm in range(n_frm):
            npsd = self.n_PSD[i_frm]
            ypsd = self.y_PSD[i_frm]

            gamma = ypsd / npsd
            xi[i_frm] = self.parameter['alpha'] * prev_xi + (1 - self.parameter['alpha']) * np.maximum(gamma - 1, 0)
            prev_xi = xi[i_frm]
        gain = np.maximum(xi/(1+xi), self.parameter['delta'])
        return gain


    def ml(self, snr):
        """
        :param y_PSD: noisy power spectral density
        :param n_PSD: noise power spectral density
        :param delta: minimum gain threshold
        :param snr : Signal to Noise Ratio (SNR)
        :param q : speech absence probability
        :return: Maximum Likelihood Envelope gain
        """
        xi = 10 ** ( snr / 10 )
        G = np.maximum(0.5 * (1 + np.sqrt(np.maximum(self.y_PSD - self.n_PSD,0) / self.y_PSD)), self.parameter['delta']);

        p = np.minimum(2 * np.sqrt(xi * self.y_PSD / self.n_PSD), 400)
        P = np.exp(-xi) * i0(p)
        Q = (self.parameter['sap'] * P) / (1 + (self.parameter['sap'] * P))

        return G * Q

    def mmse_stsa(self):
        '''
        :param y_PSD: noisy power spectral density
        :param n_PSD: noise power spectral density
        :param delta: minimum gain threshold
        :param alpha: alpha in DD approach
        :param sap: speech absent probability
        :return: MMSE-STSA gain
        '''
        n_frm, n_freq = self.y_PSD.shape
        mu = (1 - self.parameter['sap']) / self.parameter['sap']
        const_gamma = np.sqrt(np.pi) / 2
        gain = np.zeros([n_frm, n_freq])

        prev_xi = 1 ## if i_frm == 0 -> xi = self.parameter['alpha'] + (1 - self.parameter['alpha']) * np.maximum(gamma-1, 0)
        for i_frm in range(n_frm):
            npsd = self.n_PSD[i_frm]
            ypsd = self.y_PSD[i_frm]

            gamma = ypsd / npsd
            xi = self.parameter['alpha'] * prev_xi + (1 - self.parameter['alpha']) * np.maximum(gamma-1, 0)
            # if i_frm == 0:
            #     xi = self.parameter['alpha'] + (1 - self.parameter['alpha']) * np.maximum(gamma-1, 0)
            # else:
            #     xi = self.parameter['alpha'] * prev_xi + (1 - self.parameter['alpha']) * np.maximum(gamma-1, 0)

            nu = xi / (1 + xi) * gamma
            eta = xi / (1 - self.parameter['sap'])
            gain_1 = np.zeros(n_freq)
            gain_2 = np.zeros(n_freq)
            gain_final = np.zeros(n_freq)
            lambdaa = np.zeros(n_freq)

            for i_freq in range(n_freq):
                if nu[i_freq] < 600:

                    lambdaa[i_freq] = mu * np.exp(nu[i_freq]) / (1 + eta[i_freq])

                    gain_1[i_freq] = (1 + nu[i_freq]) *i0(nu[i_freq] / 2) + nu[i_freq] * i1(nu[i_freq] / 2)
                    gain_2[i_freq] = const_gamma * np.sqrt(nu[i_freq]) / gamma[i_freq] * np.exp(-nu[i_freq] / 2) * gain_1[i_freq]
                    # temp1 = i0(nu[i_freq] / 2)
                    #
                    # if temp1 > 1e+100:
                    #     print(i_freq, temp1, nu[i_freq])

                    if nu[i_freq] < 600:
                        gain_final[i_freq] = lambdaa[i_freq] / (1 + lambdaa[i_freq]) * gain_2[i_freq]
                    else:
                        gain_final[i_freq] = xi[i_freq] / (1 + xi[i_freq])

            gain[i_frm] = np.maximum(gain_final, self.parameter['delta'])
            prev_xi = gain_final * gain_final * gamma
        # print(gain_1)
        return gain


    def om_lsa(self):
        """
        :param y_PSD: noisy power spectral density
        :param n_PSD: noise power spectral density
        :param delta: minimum gain threshold
        :param alpha: alpha in DD approach
        :return: om-lsa gain
        """
        wlocal = 1
        wglobal = 15

        hlocal = hann_window(2 * wlocal + 1)
        hlocal = hlocal / sum(hlocal)
        hglobal = hann_window(2 * wglobal + 1)
        hglobal = hglobal / sum(hglobal)

        zetamin = 10 ** (-10 / 10)
        zetamax = 10 ** (-5 / 10)
        zetaratio = np.log(zetamax / zetamin)

        zetapmin = 10 ** (0 / 10)
        zetapmax = 10 ** (10 / 10)

        sapmax = 0.95
        beta = 0.7

        n_frm, n_freq = self.y_PSD.shape
        gain = np.zeros([n_frm, n_freq])
        gamma_prev = np.ones(n_freq)
        Gi = np.ones(n_freq)
        # xi_old = 1 ## if i_frm == 0 -> xi = self.parameter['alpha'] + (1 - self.parameter['alpha']) * np.maximum(gamma-1, 0)
        for i_frm in range(n_frm):
            if i_frm == 0:
                npsd = self.y_PSD[i_frm, :]
            else:
                npsd = self.n_PSD[i_frm, :]
            ypsd = self.y_PSD[i_frm, :]

            gamma = ypsd / npsd
            xi = self.parameter['alpha'] * (Gi ** 2) * gamma_prev + (1 - self.parameter['alpha']) * np.maximum(gamma-1, 0)
            nu = xi / (1 + xi) * gamma
            Gi = xi / (1 + xi) * np.exp(exp1(nu) / 2)

            if i_frm == 0:
                zeta = xi
                zetaframe = np.mean(zeta)
                zetapeak = zetaframe
                sap = sapmax * np.ones(n_freq)
            else:
                zeta = beta * zetaprev + (1 - beta) * xi_prev
                zetalocal = np.convolve(zeta, hlocal,'same')
                zetaglobal = np.convolve(zeta, hglobal, 'same')
                zetaframe = np.mean(zeta)

                mu = np.maximum(np.minimum(np.log(zetaframe / zetapeak / zetamin) / zetaratio, 1), 0)
                Plocal = np.maximum(np.minimum(np.log(zetalocal / zetamin) / zetaratio, 1), 0)
                Pglobal = np.maximum(np.minimum(np.log(zetaglobal / zetamin) / zetaratio, 1), 0)
                if zetaframe > zetamin:
                    if zetaframe > zetaframeprev:
                        Pframe = 1
                        zetapeak = np.minimum(np.maximum(zetaframe, zetapmin), zetapmax)
                    else:
                        Pframe = mu
                else:
                    Pframe = 0

                sap = np.minimum(1 - Pframe * Plocal * Pglobal, sapmax)

            spp = 1 / (1 + sap /(1 - sap) * (1 + xi) * np.exp(-nu))

            ## gain estimation
            gain_i_frm = xi / (1 + xi) * np.exp(exp1(nu)/2)
            gain[i_frm, :] = np.minimum((gain_i_frm ** spp) * (self.parameter['delta'] ** (1 - spp)), 1)

            ##finalization
            # xi_old = (gain_i_frm ** 2) * gamma
            gamma_prev = gamma
            xi_prev = xi
            zetaframeprev = zetaframe
            zetaprev = zeta

        return gain