import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import *
from utils import ensure_file

class CaptchaDataset(Dataset):
    def __init__(self, transform=None):
        parquet_path = ensure_file(
            local_path=LOCAL_PARQUET,
            repo_id=HF_REPO,
            filename=HF_FILENAME,
        )

        df = pd.read_parquet(parquet_path)
        images = np.stack(df["image"].values)          # (N, 4096)
        self.images = images.reshape(-1, 32, 128).astype(np.float32) / 255.0
    
        self.labels = df["label"]      # StringArray  ← 现在是字符串
        self.transform = transform

        self.h = IMG_H
        self.w = IMG_W

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1. 还原图片
        img = self.images[idx]  
        
        img = torch.from_numpy(img).unsqueeze(0)        # (1, 32, 128)

        if self.transform:
            img = self.transform(img)

        # 2. 标签（string → 整数序列）
        label_str = self.labels[idx]
        # 转成字符索引列表
        label_indices = [CHAR2IDX[c] for c in label_str]
        label = torch.tensor(label_indices, dtype=torch.long)

        # 返回真实长度（方便 CTC 处理变长）
        return img, label, len(label_indices)


def ctc_collate_fn(batch):
    imgs, labels, label_lengths = zip(*batch)

    imgs = torch.stack(imgs, dim=0)                       # [B, 1, H, W]

    labels = torch.cat(labels, dim=0)                     # 把所有标签拼成一维
    label_lengths = torch.tensor(label_lengths, dtype=torch.long)

    return imgs, labels, label_lengths

if __name__ == '__main__':
    dataset = CaptchaDataset()
    print(dataset.images.shape)
    print(dataset.images.dtype)
    print(len(dataset))