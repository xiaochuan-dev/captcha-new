import os

import numpy as np
import torch
from torch.utils.data import Dataset
from huggingface_hub import hf_hub_download

from config import (
    IMG_H,
    IMG_W,
    HF_REPO,
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

    def __init__(
        self,
        images_path: str | None = None,
        labels_path: str | None = None,
        transform=None,
    ):

        images_path = images_path or LOCAL_IMAGES
        labels_path = labels_path or LOCAL_LABELS

        # =========================
        # 自动下载图片
        # =========================

        images_path = ensure_file(
            local_path=images_path,
            repo_id=HF_REPO,
            filename=HF_IMAGES_FILENAME,
        )

        # =========================
        # 自动下载标签
        # =========================

        labels_path = ensure_file(
            local_path=labels_path,
            repo_id=HF_REPO,
            filename=HF_LABELS_FILENAME,
        )

        print(
            f"加载图片: {images_path}",
            flush=True,
        )

        print(
            f"加载标签: {labels_path}",
            flush=True,
        )

        # =========================
        # mmap
        # =========================

        self.images = np.load(
            images_path,
            mmap_mode="r",
        )

        self.labels = np.load(
            labels_path,
            mmap_mode="r",
        )

        self.transform = transform

        # =========================
        # 检查
        # =========================

        if self.images.ndim != 3:
            raise ValueError(
                f"图片维度错误: {self.images.shape}"
            )

        if self.images.shape[1] != IMG_H:
            raise ValueError(
                f"图片高度错误: "
                f"{self.images.shape[1]} != {IMG_H}"
            )

        if self.images.shape[2] != IMG_W:
            raise ValueError(
                f"图片宽度错误: "
                f"{self.images.shape[2]} != {IMG_W}"
            )

        if self.images.dtype != np.uint8:
            raise ValueError(
                f"图片 dtype 错误: "
                f"{self.images.dtype}"
            )

        if self.labels.ndim != 2:
            raise ValueError(
                f"标签维度错误: {self.labels.shape}"
            )

        if self.labels.shape[1] != 4:
            raise ValueError(
                f"标签长度错误: "
                f"{self.labels.shape[1]} != 4"
            )

        if self.labels.dtype != np.uint8:
            raise ValueError(
                f"标签 dtype 错误: "
                f"{self.labels.dtype}"
            )

        if len(self.images) != len(self.labels):
            raise ValueError(
                "图片和标签数量不一致: "
                f"{len(self.images)} vs "
                f"{len(self.labels)}"
            )

        print(
            f"图片 shape: {self.images.shape}",
            flush=True,
        )

        print(
            f"图片 dtype: {self.images.dtype}",
            flush=True,
        )

        print(
            f"标签 shape: {self.labels.shape}",
            flush=True,
        )

        print(
            f"标签 dtype: {self.labels.dtype}",
            flush=True,
        )

        print(
            f"样本数: {len(self):,}",
            flush=True,
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        if idx < 0:
            idx += len(self)

        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        # =========================
        # 图片
        # =========================

        # mmap -> numpy -> torch
        image = torch.from_numpy(
            self.images[idx].copy()
        )

        # uint8 [0,255]
        # ->
        # float32 [0,1]
        image = image.float().div_(255.0)

        # [32,128]
        # ->
        # [1,32,128]
        image = image.unsqueeze(0)

        if self.transform is not None:
            image = self.transform(image)

        # =========================
        # 标签
        # =========================

        label = torch.from_numpy(
            self.labels[idx].copy()
        ).long()

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