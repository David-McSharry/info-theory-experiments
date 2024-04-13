import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import scipy.io
from torch.utils.data import TensorDataset, DataLoader
from scipy.signal import butter, filtfilt, decimate
from scipy.fftpack import fft, fftfreq
from sklearn.decomposition import PCA


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def estimate_MI_smile(scores):
    """
    Returns the MI estimate using the SMILE estimator given the scores matrix and a clip
    """
    clip = 5
    
    first_term = scores.diag().mean()

    batch_size = scores.size(0)

    # clip scores between -clip and clip
    clipped_scores = torch.clamp(scores, -clip, clip)
    clipped_scores = scores

    # e^clipped_scores
    exp_clipped_scores = torch.exp(clipped_scores)

    mask = (torch.ones_like(exp_clipped_scores).to(device) - torch.eye(batch_size).to(device))

    masked_exp_clipped_scores = exp_clipped_scores * mask

    num_non_diag = mask.sum()

    mean_exp_clipped_scores = masked_exp_clipped_scores.sum() / num_non_diag

    second_term = torch.log2(mean_exp_clipped_scores)

    return (1/torch.log(torch.tensor(2.0))) * first_term - second_term


# def fft_check(data, fs):
#     # Perform FFT on your data and plot results to view frequencies present in data
#     N = data.shape[0]
#     yf = fft(data)
#     xf = fftfreq(N, 1/fs)

#     plt.figure(figsize=(10,6))
#     plt.plot(xf, np.abs(yf))

#     plt.title('FFT of data')
#     plt.xlabel('Frequency (Hz)')
#     plt.ylabel('Magnitude')
#     plt.grid()
#     plt.xscale('log')
#     plt.show()

#     return None


# https://neurotycho.brain.riken.jp/download/2012/20100802S1_Epidural-ECoG+Food-Tracking_B_Kentaro+Shimoda_mat_ECoG64-Motion6.zip


def prepare_batch(X):
    # take all samples except the last one as inputs
    input_data = X[:-1]

    # take all samples except the first one as targets
    target_data = X[1:]

    # stack them as pairs with dimension (999, 2, 10)
    pairs = torch.stack((input_data, target_data), dim=1)
    
    assert pairs[0,0,0] == input_data[0,0]

    return pairs


def prepare_ecog_dataset():


    def _butter_highpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return b, a


    def _butter_highpass_filter(data, cutoff, fs, order=5):
        b, a = _butter_highpass(cutoff, fs, order=order)
        y = filtfilt(b, a, data)
        return y

    
    num_channels = 64
    data_list = []
    for i in range(1, num_channels + 1):

        channel_data = scipy.io.loadmat(f'/vol/bitbucket/dm2223/info-theory-experiments/data/ecog_data_raw/ECoG_ch{i}.mat')
        data = channel_data[f'ECoGData_ch{i}'].squeeze()

        # high pass filter at 1 Hz
        fs = 1000  # original sampling rate
        cutoff = 1  # cutoff frequency for high pass filter
        data = _butter_highpass_filter(data, cutoff, fs)

        # downsample to 300 Hz
        downsample_rate = int(fs / 300)  # calculate downsample rate
        data = decimate(data, downsample_rate)

        # standardise across features. Comes last as the data being fed into our ML model should be standardised
        data = (data - np.mean(data)) / np.std(data)

        data_list.append(data) 

    # Stack all channel data into a single numpy array
    all_data = np.stack(data_list, axis=0)

    dataset_tensor = torch.tensor(all_data).T

    dataset_pairs = prepare_batch(dataset_tensor)
    # save as dataset
    torch.save(dataset_pairs, '/vol/bitbucket/dm2223/info-theory-experiments/data/ecog_data_pairs.pth')

    return None




