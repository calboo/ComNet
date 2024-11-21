
import torch
import numpy as np
import pandas as pd
import os,shutil 
from torch.utils import data


class Dataset(data.Dataset):
    
    def __init__(self, input_path, target_path, startdate = None, enddate = None,
                 history_length = 100, sample_noise = 0, target_noise = 0):
        
        self.input_path = input_path
        self.target_path = target_path
        self.history_len = history_length
        
        self.input_data = np.load(input_path, allow_pickle=True) 
        self.target_data = np.load(target_path, allow_pickle=True)

        self.sample_noise = sample_noise
        self.target_noise = target_noise
        
        if startdate:
            
            startdate = pd.to_datetime(startdate,format='%Y-%m-%d')
            index_start = np.where(self.input_data.index >= startdate)[0][0]
            
            if index_start >= history_length+1:
                self.input_data = self.input_data.iloc[index_start-history_length+1:]
                self.target_data = self.target_data.iloc[index_start-history_length+1:]

        if enddate:
            
            enddate = pd.to_datetime(enddate,format='%Y-%m-%d')
            self.input_data = self.input_data[self.input_data.index < enddate]
            self.target_data = self.target_data[self.target_data.index < enddate]
            
        self.input_tensor = torch.tensor(self.input_data.values, dtype=torch.float32)
        self.target_tensor = torch.tensor(self.target_data.values, dtype=torch.float32)

    def __len__(self):
        
        return len(self.input_data)-self.history_len
    
    def __getitem__(self, index):  
        
        sample = self.input_tensor[index:index+self.history_len]
        target = self.target_tensor[index+self.history_len-1]

        if self.sample_noise > 0:
            gaussian_noise = np.float32(np.random.normal(0, self.sample_noise, sample.shape))
            sample = sample * (1.0 + gaussian_noise)

        return sample, target

    