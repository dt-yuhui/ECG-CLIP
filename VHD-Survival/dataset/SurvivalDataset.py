import os
import math
import struct
import h5py

import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
from factory.survival import make_surv_array
from factory.ecg import bandpass_filtering, notch_filtering, resampling, zero_padding


def read_dat(file_path):
    f = open(file_path, "rb")
    ecg_signal = []
    for i in range(8):
        lead_data = []
        for pos in range(0, 500 * 10):
            b = f.read(2)
            value = struct.unpack("h", b)[0]
            lead_data.append(value)
        ecg_signal.append(np.array(lead_data))
    """
    ecg_signal.insert(2, ecg_signal[1] - ecg_signal[0])
    ecg_signal.insert(3, -0.5 * (ecg_signal[1] + ecg_signal[0]))
    ecg_signal.insert(4, ecg_signal[0] - 0.5 * ecg_signal[1])
    ecg_signal.insert(5, ecg_signal[1] - 0.5 * ecg_signal[0])
    """
    ecg_signal = np.vstack(ecg_signal) * 0.00244
    ecg_signal = np.transpose(ecg_signal)

    ecg_signal = bandpass_filtering(ecg_signal, 500, 0.5, 100)
    ecg_signal = notch_filtering(ecg_signal, 500, 50)
    ecg_signal = resampling(ecg_signal, 500, 400, method='polyphase')
    ecg_signal = zero_padding(ecg_signal)

    ecg_signal = np.transpose(ecg_signal)

    return ecg_signal


class SurvivalDataset(Dataset):
    def __init__(self, phase, label_csv, label, h5_path=None, h5_csv_path=None, dat_dir=None):
        super(SurvivalDataset, self).__init__()
        self.phase = phase
        self.label = label
        self.h5_path = h5_path
        self.h5_csv_path = h5_csv_path
        self.dat_dir = dat_dir
        self.df = pd.read_csv(label_csv, low_memory=False)

        self.set_index(h5_csv_path=self.h5_csv_path)
        breaks = np.arange(0., 365. * 5, 365. / 8)
        self.labels = make_surv_array(self.df['fu_days'], self.df[label], breaks)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        if self.h5_path and self.h5_csv_path:
            with h5py.File(self.h5_path, 'r') as dataset_file:
                ecg_signal = np.array(dataset_file['ECGs'][int(row['h5_index'])])
        else:
            file_path = os.path.join(self.dat_dir, row['Filename'] + ".dat")
            ecg_signal = read_dat(file_path)

        ecg_signal = torch.from_numpy(ecg_signal).float()
        label = torch.from_numpy(self.labels[idx]).float()
        return ecg_signal, label

    def __len__(self):
        return len(self.df)

    def set_index(self, h5_csv_path):
        df_h5 = pd.read_csv(h5_csv_path, low_memory=False)
        df_h5['h5_index'] = df_h5.index
        print(f'h5 len: {len(df_h5)}; label len: {len(self.df)}')
        self.df = pd.merge(df_h5, self.df, on='Filename', how='inner')
        print('merge', len(self.df))
