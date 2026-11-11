
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

        # 直接保留 bytes，不提前转换全部图片
        self.images = df["image"].values
        self.labels = df["label"].values

        self.transform = transform

        self.h = IMG_H
        self.w = IMG_W

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1. bytes -> uint8 -> float32
        img = np.frombuffer(self.images[idx], dtype=np.uint8)

        if img.size != self.h * self.w:
            raise ValueError(
                f"图片尺寸错误: index={idx}, "
                f"实际像素={img.size}, "
                f"预期像素={self.h * self.w}"
            )

        img = img.reshape(self.h, self.w).astype(np.float32) / 255.0

        img = torch.from_numpy(img).unsqueeze(0)

        if self.transform:
            img = self.transform(img)

        # 2. 标签转整数序列
        label_str = self.labels[idx]
        label_indices = [CHAR2IDX[c] for c in label_str]

        label = torch.tensor(label_indices, dtype=torch.long)

        return img, label, len(label_indices)


def ctc_collate_fn(batch):
    imgs, labels, label_lengths = zip(*batch)

    imgs = torch.stack(imgs, dim=0)
    labels = torch.cat(labels, dim=0)
    label_lengths = torch.tensor(label_lengths, dtype=torch.long)

    return imgs, labels, label_lengths


if __name__ == '__main__':
    dataset = CaptchaDataset()

    print("图片数量:", len(dataset))
    print("图片数据类型:", type(dataset.images[0]))
    print("图片字节数:", len(dataset.images[0]))

    img, label, label_length = dataset[0]

    print("图片形状:", img.shape)
    print("图片 dtype:", img.dtype)
    print("标签:", label)
    print("标签长度:", label_length)
