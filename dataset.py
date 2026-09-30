import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from huggingface_hub import hf_hub_download

from config import *

def ensure_file(
    local_path: str,
    repo_id: str,
    filename: str,
) -> str:
    """
    本地没有则从 Hugging Face 自动下载。
    """
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        print(f"使用本地数据: {local_path}", flush=True)
        return local_path

    print(f"本地未找到 {local_path}", flush=True)
    print(f"正在从 Hugging Face 下载: {filename}", flush=True)
    print(f"  repo: {repo_id}", flush=True)

    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)

    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=os.path.dirname(os.path.abspath(local_path)) or ".",
    )

    if os.path.abspath(downloaded) != os.path.abspath(local_path):
        if not os.path.exists(local_path):
            os.replace(downloaded, local_path)

    print(f"下载完成: {local_path}", flush=True)
    return local_path


class CaptchaDataset(Dataset):
    def __init__(self, transform=None):
        parquet_path = ensure_file(
            local_path=LOCAL_PARQUET,
            repo_id=HF_REPO,
            filename=HF_FILENAME,
        )

        df = pd.read_parquet(parquet_path)
        images = np.stack(df["image"].values)          # (N, 4096)
        self.images = images.reshape(-1, 32, 128)      # (N, 32, 128)
    
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