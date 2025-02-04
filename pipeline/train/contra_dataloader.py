import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random

class SupervisedDataset(Dataset):
    def __init__(self, database):
        self.data = database
        self.captions = captions

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        video_path = entry["video_path"]
        query = entry["query"]  
        gt_caption = entry["caption"] 
 
        return video_path, query, gt_caption