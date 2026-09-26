import os
import pyarrow.parquet as pq
import numpy as np
import torch
from torch.utils.data import Dataset
from huggingface_hub import hf_hub_download

from config import (
    IMG_H,
    IMG_W,
    HF_REPO,
    LOCAL_PARQUET,
    HF_FILENAME
)


# =========================
# 数据集文件
# =========================

HF_IMAGES_FILENAME = "images.npy"
HF_LABELS_FILENAME = "labels.npy"

LOCAL_IMAGES = "./data/images.npy"
LOCAL_LABELS = "./data/labels.npy"


def ensure_file(
    local_path: str,
    repo_id: str,
    filename: str,
) -> str:
    """
    本地没有则从 Hugging Face 自动下载。
    """

    if (
        os.path.exists(local_path)
        and os.path.getsize(local_path) > 0
    ):
        print(
            f"使用本地数据: {local_path}",
            flush=True,
        )
        return local_path

    print(
        f"本地未找到 {local_path}",
        flush=True,
    )

    print(
        f"正在从 Hugging Face 下载: {filename}",
        flush=True,
    )

    print(
        f"  repo: {repo_id}",
        flush=True,
    )

    os.makedirs(
        os.path.dirname(local_path) or ".",
        exist_ok=True,
    )

    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=os.path.dirname(
            os.path.abspath(local_path)
        ) or ".",
    )

    # hf_hub_download 返回的路径可能就是 local_path
    if os.path.abspath(downloaded) != os.path.abspath(local_path):
        if not os.path.exists(local_path):
            os.replace(
                downloaded,
                local_path,
            )

    print(
        f"下载完成: {local_path}",
        flush=True,
    )

    return local_path


class CaptchaDataset(Dataset):
    def __init__(self, transform=None):

        parquet_path = ensure_file(
            local_path=LOCAL_PARQUET,
            repo_id=HF_REPO,
            filename=HF_FILENAME,
        )

        self.table = pq.read_table(parquet_path)
        self.images = self.table["image"]      # BinaryArray
        self.labels = self.table["labels"]     # FixedSizeListArray
        self.transform = transform

        self.h = IMG_H
        self.w = IMG_W

    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        # 1. 还原图片
        raw = self.images[idx].as_py()          # bytes
        img = np.frombuffer(raw, dtype=np.uint8).reshape(self.h, self.w)

        image = torch.from_numpy(img.copy()).float().div_(255.0)
        image = image.unsqueeze(0)              # [1, H, W]

        if self.transform is not None:
            image = self.transform(image)

        # 2. 标签
        label = torch.tensor(self.labels[idx].as_py(), dtype=torch.long)

        return image, label, 4

def ctc_collate_fn(batch):

    imgs, labels, label_lengths = zip(*batch)

    imgs = torch.stack(
        imgs,
        dim=0,
    )

    labels = torch.cat(
        labels,
        dim=0,
    )

    label_lengths = torch.tensor(
        label_lengths,
        dtype=torch.long,
    )

    return imgs, labels, label_lengths