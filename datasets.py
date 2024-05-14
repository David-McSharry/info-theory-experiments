from torch.utils.data import Dataset
import torch
import numpy as np
import scipy.io
import requests
import zipfile
import io
from scipy.signal import butter, filtfilt
from utils import prepare_batch

class BitStringDataset(Dataset):
    def __init__(self, gamma_parity, gamma_extra, length):
        self.data = self.generate_bit_string(gamma_parity, gamma_extra, length)

    def generate_bit_string(self, gamma_parity, gamma_extra, length):
        bit_strings = np.zeros((length, 6), dtype=np.int64)
        bit_strings[0, :] = np.random.randint(0, 2, 6)
        for t in range(1, length):
            parity = np.sum(bit_strings[t-1, :-1]) % 2
            if np.random.rand() < gamma_parity:
                bit_strings[t, :-1] = np.random.choice([0, 1], size=5)
                sum_parity = bit_strings[t, :-1].sum() % 2
                if sum_parity != parity:
                    bit_strings[t, np.random.choice(5)] ^= 1
            else:
                bit_strings[t, :-1] = np.random.choice([0, 1], size=5)
                sum_parity = bit_strings[t, :-1].sum() % 2
                if sum_parity == parity:
                    bit_strings[t, np.random.choice(5)] ^= 1
            if np.random.rand() < gamma_extra:
                bit_strings[t, -1] = bit_strings[t-1, -1]
            else:
                bit_strings[t, -1] = 1 - bit_strings[t-1, -1]

        adjacent_bits = np.array([bit_strings[i-1:i+1] for i in range(1, length)])
        return torch.tensor(adjacent_bits, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class ECoGDataset(Dataset):
    def __init__(
            self,
            prepare_pairs: bool = True,
        ):

        self.data = self._prepare_ecog_dataset_no_local(prepare_pairs)

    def __len__(self):
        return self.data.size(0)
    
    def __getitem__(self, idx):
        return self.data[idx]

    def download_and_extract(self, url):
        response = requests.get(url)
        zip_file = zipfile.ZipFile(io.BytesIO(response.content))
        return {name: zip_file.read(name) for name in zip_file.namelist()}


    def _prepare_ecog_dataset_no_local(self, prepare_pairs):

        def _butter_highpass(cutoff, fs, order=5):
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            b, a = butter(order, normal_cutoff, btype='high', analog=False)
            return b, a


        def _butter_highpass_filter(data, cutoff, fs, order=5):
            b, a = _butter_highpass(cutoff, fs, order=order)
            y = filtfilt(b, a, data)
            return y
        
        URL = 'https://neurotycho.brain.riken.jp/download/2012/20100802S1_Epidural-ECoG+Food-Tracking_B_Kentaro+Shimoda_mat_ECoG64-Motion6.zip'
        data_files = self.download_and_extract(URL)

        num_channels = 64
        data_list = []

        for i in range(1, num_channels + 1):
            file_name = f'20100802S1_Epidural-ECoG+Food-Tracking_B_Kentaro+Shimoda_mat_ECoG64-Motion6/ECoG_ch{i}.mat'
            if file_name in data_files:
                # Load MAT file from in-memory data
                mat_contents = io.BytesIO(data_files[file_name])
                channel_data = scipy.io.loadmat(mat_contents)
                data = channel_data[f'ECoGData_ch{i}'].squeeze()

                # Process data
                fs = 1000  # Sampling rate
                cutoff = 1  # Hz
                data = _butter_highpass_filter(data, cutoff, fs)
                data = data[::3] # 1000
                data = (data - np.mean(data)) / np.std(data)

                data_list.append(data)
            else:
                raise FileNotFoundError(f"File {file_name} not found in the downloaded data")

        all_data = np.stack(data_list, axis=0)
        
        dataset_tensor = torch.tensor(all_data).T

        # here prepare_batch needs to be predefined or implemented
        if prepare_pairs:
            dataset_pairs = prepare_batch(dataset_tensor)
            return dataset_pairs
        else:
            return dataset_tensor

