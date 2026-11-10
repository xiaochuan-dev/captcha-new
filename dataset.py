import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from config import (
    IMG_H, IMG_W, CHAR2IDX,
    LOCAL_PARQUET, HF_REPO, HF_FILENAME,
    VAL_RATIO,
)
from utils import ensure_file


class CaptchaDataset(Dataset):
    def __init__(self, transform=None, indices=None, images=None, labels=None):
        self.transform = transform

        if images is None or labels is None:
            # 只在第一次真正加载数据
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

        # 关键：保证 indices 是 int64 的 numpy 数组
        if indices is None:
            self.indices = np.arange(len(images), dtype=np.int64)
        else:
            self.indices = np.asarray(indices, dtype=np.int64)

    def set_transform(self, transform):
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = int(self.indices[idx])          # 强制转成 python int，避免类型问题

        img = self.images[real_idx]                # (H, W)
        img = torch.from_numpy(img).unsqueeze(0)   # (1, H, W)

        if self.transform is not None:
            img = self.transform(img)

        label_str = self.labels[real_idx]
        label_indices = [CHAR2IDX[c] for c in label_str]
        label = torch.tensor(label_indices, dtype=torch.long)

        return img, label, len(label_indices)


def get_datasets(train_transform=None, seed=42):
    """
    底层数据只加载一次，返回 train / val 两个数据集
    """
    full = CaptchaDataset(transform=None)

    dataset_size = len(full)
    val_size = max(1, int(VAL_RATIO * dataset_size))
    train_size = dataset_size - val_size

    rng = np.random.RandomState(seed)
    all_indices = rng.permutation(dataset_size).astype(np.int64)

    train_indices = all_indices[:train_size]
    val_indices   = all_indices[train_size:]

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