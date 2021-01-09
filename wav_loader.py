#coding: utf-8
#Author: ltz
#Date: 2021.1.8

from torch.utils.data import Dataset, DataLoader 
import librosa as rosa
import torch 
import os 
import numpy as np
import glob 

def _load_wav( file_name, frame_dur, sr=16000 ):
    if file_name is None:
        return None
    pcm_data, _ = rosa.load( file_name, sr )
    win_len = int( frame_dur*sr/1000 )

    return torch.tensor(pcm_data)

class WavDataset(Dataset):
    def __init__(self, noisy_paths, clean_paths, frame_dur=37.5):
        self.noisy_paths = noisy_paths
        self.clean_paths = clean_paths
        self.loader = _load_wav
        self.frame_dur = frame_dur

    def __getitem__(self, item):
        noisy_file = self.noisy_paths[item]
        clean_file = self.clean_paths[item]
        return self.loader(noisy_file, self.frame_dur), self.loader(clean_file, self.frame_dur)

    def __len__(self):
        return len(self.noisy_paths)

def get_all_file_name( dns_home_path ):
    noisy_path = os.path.join( dns_home_path, "noisy")
    clean_path = os.path.join( dns_home_path, "clean")

    noise_files = glob.glob( noisy_path + '/*.wav' )

    noisy_paths = []
    clean_paths = []
    for absName in noise_files:
        noisy_paths.append( absName )
        file_name = os.path.basename(absName)
        code = str(file_name).split('_')[-1]
        clean_name = os.path.join(dns_home_path, 'clean', 'clean_fileid_%s' % code)
        #clean_name = os.path.join( clean_path, file_name )
        clean_paths.append( clean_name )
    
    return noisy_paths, clean_paths