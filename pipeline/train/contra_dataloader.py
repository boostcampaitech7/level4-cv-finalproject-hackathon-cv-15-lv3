import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Supervised Dataset
class SupervisedDataset(Dataset):
    def __init__(self, gt_data, faiss_search):
        self.data = gt_data
        self.faiss_search = faiss_search

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        gt_caption = entry["caption"]

        # Top-K 검색
        top_k_results = self.faiss_search.find_similar_captions(gt_caption)
        
        positive_sample = top_k_results[0][0]  # Top-1
        negative_sample = random.choice([res[0] for res in top_k_results[1:]])  # Top-K 중 1개

        return gt_caption, positive_sample, negative_sample