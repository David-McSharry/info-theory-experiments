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


class ResidualStreamDataset(Dataset):
    def __init__(
        self,
        AI_gen: bool,
        index: int
    ):
        if AI_gen  == True:
            data_path = f"activations/resid_actications_layer_{index}.pt"
            data = torch.load(data_path)
        else:
            data_path = f"activations/econ_resid_activations_layer_{index}.pt"
            data = torch.load(data_path)
    
        self.data = prepare_batch(data)
        
    def __len__(self):
        return self.data.size(0)
    
    def __getitem__(self, idx):
        return self.data[idx]


            

class FMRIDatasetConcat(Dataset):
    """
    Still valid, less data
    """
    def __init__(self):
        self.indices = list(range(37, 50)) + list(range(89, 100))
        self.data = prepare_batch(self.load_data())

    def load_data(self):
        concat_data = []
        all_data = scipy.io.loadmat('HCP_100subj/Schaefer100_BOLD_HCP.mat')
        all_data = all_data['BOLD_timeseries_HCP']
        for index in self.indices:
            data_for_index = all_data[index][0]
            concat_data.append(data_for_index)
        concat_data = np.concatenate(concat_data, axis=1)
        return torch.tensor(concat_data, dtype=torch.float32).T

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx]



class FMRIDatasetConcatNoPrepareBatch(Dataset): # this is shape [N,D] rahter than [N-1,2,D]
    """
    Still valid, less data
    """
    def __init__(self):
        self.indices = list(range(37, 50)) + list(range(89, 100))
        self.data = self.load_data()

    def load_data(self):
        concat_data = []
        all_data = scipy.io.loadmat('HCP_100subj/Schaefer100_BOLD_HCP.mat')
        all_data = all_data['BOLD_timeseries_HCP']
        for index in self.indices:
            data_for_index = all_data[index][0]
            concat_data.append(data_for_index)
        concat_data = np.concatenate(concat_data, axis=1)
        return torch.tensor(concat_data, dtype=torch.float32).T

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx]
    

class FMRIDatasetConcatV2(Dataset):
    """
    The old version messed up indices in a way that does not invalidate results, just means they were produced on less data

    This version uses the indices properly to only use default brain network.
    """
    def __init__(self):
        data = scipy.io.loadmat('HCP_100subj/Schaefer100_BOLD_HCP.mat')
        data_concat = []
        for i in range(100):
            patient_data = data['BOLD_timeseries_HCP'][i][0]
            data_concat.append(patient_data)
        data_concat = np.concatenate(data_concat, axis=1).T
        indices = list(range(37, 50)) + list(range(89, 100))
        num_indices = len(indices)
        new_dataset = np.zeros((data_concat.shape[0], num_indices))
        for i, idx in enumerate(indices):
            new_dataset[:, i] = data_concat[:, idx]
        self.data = torch.tensor(new_dataset, dtype=torch.float32)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx]


class BigActDataset(Dataset):
    def __init__(self):
        all_data = []
        for j in range(1, 15):
            data = torch.load(f"activations2/econ_resid_actications_layer_3_{j}.pt")
            all_data.append(data)
        concatenated_data = torch.cat(all_data, dim=0)
        formatted_data = prepare_batch(concatenated_data)
        self.data = (formatted_data - formatted_data.mean(dim=0)) / formatted_data.std(dim=0)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx]


class MegDataset(Dataset):
    def __init__(self):
        # Load the .mat file
        data_pla = scipy.io.loadmat('070814_2_PLA (1).mat')
        
        # Concatenate the data along the second dimension
        concatenated_data = np.concatenate([data_pla['timeseries'][0][i].flatten() for i in range(data_pla['timeseries'][0].shape[0])], axis=0)
        
        # Reshape the concatenated data to the desired shape
        reshaped_data = concatenated_data.reshape(data_pla['timeseries'][0].shape[0], -1)

        # Convert to a torch tensor
        data = (torch.tensor(reshaped_data, dtype=torch.float32).T)

        # standardize the data
        data = (data - data.mean(dim=0)) / data.std(dim=0)

        self.data = prepare_batch(data)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx]


