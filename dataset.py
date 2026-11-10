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
    def __init__(self, transform=None, indices=None, images=None, labels=None):
        """
        indices: 如果传入，表示这是一个子集
        images / labels: 用于共享底层数据，避免重复加载
        """
        self.transform = transform

        if images is None or labels is None:
            # 第一次创建时才真正加载数据
            parquet_path = ensure_file(
                local_path=LOCAL_PARQUET,
                repo_id=HF_REPO,
                filename=HF_FILENAME,
            )
            print(f"使用本地数据: {parquet_path}", flush=True)

            df = pd.read_parquet(parquet_path)
            images = np.stack(df["image"].values).reshape(-1, IMG_H, IMG_W).astype(np.float32) / 255.0
            labels = df["label"].tolist()

        self.images = images
        self.labels = labels
        self.indices = indices if indices is not None else np.arange(len(images))

    def set_transform(self, transform):
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]

        img = self.images[real_idx]                          # (H, W)
        img = torch.from_numpy(img).unsqueeze(0)             # (1, H, W)

        if self.transform is not None:
            img = self.transform(img)

        label_str = self.labels[real_idx]
        label_indices = [CHAR2IDX[c] for c in label_str]
        label = torch.tensor(label_indices, dtype=torch.long)

        return img, label, len(label_indices)


def get_datasets(train_transform=None, seed=42):
    """
    一次性加载数据，并返回 train_dataset 和 val_dataset。
    底层 images / labels 只存在一份，内存不翻倍。
    """
    # 先加载完整数据
    full = CaptchaDataset(transform=None)

    dataset_size = len(full)
    val_size = max(1, int(VAL_RATIO * dataset_size))
    train_size = dataset_size - val_size

    rng = np.random.RandomState(seed)
    indices = rng.permutation(dataset_size)

    train_indices = indices[:train_size]
    val_indices   = indices[train_size:]

    # 创建两个视图，共享同一份 images 和 labels
    train_dataset = CaptchaDataset(
        transform=train_transform,
        indices=train_indices,
        images=full.images,
        labels=full.labels,
    )
    val_dataset = CaptchaDataset(
        transform=None,
        indices=val_indices,
        images=full.images,
        labels=full.labels,
    )

    print(f"Train: {train_size:,} | Val: {val_size:,}", flush=True)
    return train_dataset, val_dataset


def ctc_collate_fn(batch):
    imgs, labels, label_lengths = zip(*batch)

    imgs = torch.stack(imgs, dim=0)
    labels = torch.cat(labels, dim=0)
    label_lengths = torch.tensor(label_lengths, dtype=torch.long)

    return imgs, labels, label_lengths