#!/usr/bin/env python
# coding=utf-8
# Author: Shuai Tao
# Mail: stao@create.aau.dk
# Create Time: Tue 14 Feb 2023 07:28:49 PM CET

import numpy as np
from scipy.signal import hanning
from scipy.fftpack import fft

import numpy as np


def init_noise_tracker_ideal_vad(noisy, fr_size, fft_size, hop, sq_hann_window):

    noisy_dft_frame_matrix = np.zeros((fft_size, 5), dtype=complex)
    for I in range(1):
        noisy_frame = sq_hann_window * noisy[(I * hop):(I * hop + fr_size)]
        noisy_dft_frame_matrix[:, I] = np.fft.fft(noisy_frame, fft_size)
    noise_psd_init = np.mean(np.abs(noisy_dft_frame_matrix[:fr_size // 2 + 1, :]) ** 2, axis=1)

    return noise_psd_init


def noisepowproposed_spp(noisy, fs, SPP):
    fr_len = int(16e-3 * fs)
    f_shift = fr_len // 2
    n_frames = int(np.floor((len(noisy) - fr_len) / f_shift)) + 1

    an_win = hanning(fr_len, sym=False)

    noise_pow_mat = np.zeros((fr_len // 2 + 1, n_frames))

    noise_pow = init_noise_tracker_ideal_vad(noisy, fr_len, fr_len, f_shift, an_win)
    # noise_pow_mat[:, 0] = noise_pow

    PH1_mean = 0.5
    alpha_PH1_mean = 0.9
    alpha_PSD = 0.8

    q = 0.5
    prior_fact = q / (1 - q)
    xi_opt_db = 15
    xi_opt = 10 ** (xi_opt_db / 10)
    log_GLR_fact = np.log(1 / (1 + xi_opt))
    GLR_exp = xi_opt / (1 + xi_opt)

    for ind_fr in range(n_frames):
        indices = ind_fr * f_shift + np.arange(fr_len)
        noisy_frame = an_win * noisy[indices]
        noisy_dft_frame = fft(noisy_frame)
        noisy_dft_frame = noisy_dft_frame[:fr_len // 2 + 1]

        noisy_per = np.abs(noisy_dft_frame) ** 2
        snr_post1 = noisy_per / noise_pow

        # GLR = prior_fact * np.exp(np.minimum(log_GLR_fact + GLR_exp * snr_post1, 200))
        # PH1 = GLR / (1 + GLR)
        #
        # PH1_mean = alpha_PH1_mean * PH1_mean + (1 - alpha_PH1_mean) * PH1
        # stuck_ind = PH1_mean > 0.99
        # PH1[stuck_ind] = np.minimum(PH1[stuck_ind], 0.99)

        PH1 = SPP[:, ind_fr]

        estimate = PH1 * noise_pow + (1 - PH1) * noisy_per
        # noise_pow = alpha_PSD * noise_pow + (1 - alpha_PSD) * estimate
        noise_pow = estimate
        noise_pow_mat[:, ind_fr] = noise_pow

    return noise_pow_mat, snr_post1


