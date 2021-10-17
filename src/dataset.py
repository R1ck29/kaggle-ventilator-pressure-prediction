import torch
from torch.utils.data import Dataset
import numpy as np
# ====================================================
# Dataset
# ====================================================
class TrainDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.df = df
        self.groups = df.groupby('breath_id').groups
        self.keys = list(self.groups.keys())
        
    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        indexes = self.groups[self.keys[idx]]
        df = self.df.iloc[indexes]
        # cate_seq_x = torch.LongTensor(df[cfg.cate_seq_cols].values)
        cont_seq_x = torch.FloatTensor(df[self.cfg.cont_seq_cols].values)
        # u_out = torch.LongTensor(df['u_out'].values)
        # y = np.float32(df["pressure"]).reshape(-1, 80, 1)
        label = torch.FloatTensor(np.float32(df["pressure"]).reshape(-1, 80, 1))
        return cont_seq_x, label
    

class TestDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.groups = df.groupby('breath_id').groups
        self.keys = list(self.groups.keys())
        
    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        indexes = self.groups[self.keys[idx]]
        df = self.df.iloc[indexes]
        # cate_seq_x = torch.LongTensor(df[cfg.cate_seq_cols].values)
        cont_seq_x = torch.FloatTensor(df[self.cfg.cont_seq_cols].values)
        return cont_seq_x