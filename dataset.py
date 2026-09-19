"""
从 Hugging Face parquet 加载验证码数据集。
自动下载；可变尺寸 RGB 位图 → 灰度 → 等比缩放 + 居中填充到 32x128。
"""
import os
import shutil

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from huggingface_hub import hf_hub_download

from config import (
    CHAR2IDX,
    IMG_H,
    IMG_W,
    HF_REPO,
    HF_FILENAME,
    LOCAL_PARQUET,
)


def ensure_parquet(
    local_path: str = LOCAL_PARQUET,
    repo_id: str = HF_REPO,
    filename: str = HF_FILENAME,
) -> str:
    """本地没有则从 Hugging Face 自动下载。"""
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        print(f"使用本地数据集: {local_path}")
        return local_path

    print(f"本地未找到 {local_path}，正在从 Hugging Face 下载...")
    print(f"  repo: {repo_id}")
    print(f"  file: {filename}")

    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)

    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=os.path.dirname(os.path.abspath(local_path)) or ".",
    )

    # hf_hub_download 可能放到子目录，统一到期望路径
    if os.path.abspath(downloaded) != os.path.abspath(local_path):
        if not os.path.exists(local_path):
            shutil.copy2(downloaded, local_path)
        print(f"已保存到: {local_path}")

    return local_path


def bytes_to_rgb(image_bytes: bytes, height: int, width: int) -> np.ndarray:
    """parquet 中的 RGB 位图字节 → (H, W, 3) uint8"""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    return arr.reshape(int(height), int(width), 3)


def resize_keep_ratio_pad(
    img: Image.Image,
    target_h: int = IMG_H,
    target_w: int = IMG_W,
    fill: int = 255,
) -> Image.Image:
    """
    等比缩放后居中填充到固定尺寸。
    不同原始宽高比的验证码都能对齐到模型输入，避免直接拉伸变形。
    """
    src_w, src_h = img.size
    if src_w <= 0 or src_h <= 0:
        return Image.new(img.mode, (target_w, target_h), fill)

    scale = min(target_w / src_w, target_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))

    img = img.resize((new_w, new_h), Image.BILINEAR)

    canvas = Image.new(img.mode, (target_w, target_h), fill)
    offset_x = (target_w - new_w) // 2
    offset_y = (target_h - new_h) // 2
    canvas.paste(img, (offset_x, offset_y))
    return canvas


def rgb_to_model_input(
    rgb: np.ndarray,
    h: int = IMG_H,
    w: int = IMG_W,
) -> torch.Tensor:
    """
    RGB (H,W,3) uint8
      → 灰度
      → 等比缩放 + 白底居中填充到 (h, w)
      → float32 [0, 1]，形状 (1, h, w)
    """
    img = Image.fromarray(rgb, mode="RGB").convert("L")
    img = resize_keep_ratio_pad(img, target_h=h, target_w=w, fill=255)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # (h, w)
    return torch.from_numpy(arr).unsqueeze(0)  # (1, h, w)


class CaptchaDataset(Dataset):
    def __init__(self, parquet_path: str | None = None, transform=None):
        path = ensure_parquet(parquet_path or LOCAL_PARQUET)
        print(f"加载数据集: {path}")
        self.df = pd.read_parquet(path)
        self.transform = transform
        print(f"样本数: {len(self.df):,}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        rgb = bytes_to_rgb(row["image"], row["height"], row["width"])
        img = rgb_to_model_input(rgb)  # (1, 32, 128)

        if self.transform is not None:
            img = self.transform(img)

        # 大写映射为小写，词表只有 0-9a-z
        label_str = str(row["label"]).lower()
        indices = [CHAR2IDX[c] for c in label_str if c in CHAR2IDX]
        if not indices:
            # 极端情况：标签为空或全是未知字符，给一个 dummy 避免 CTC 崩
            indices = [1]
        label = torch.tensor(indices, dtype=torch.long)

        return img, label, len(label)


def ctc_collate_fn(batch):
    imgs, labels, label_lengths = zip(*batch)
    imgs = torch.stack(imgs, 0)
    labels = torch.cat(labels, dim=0)
    label_lengths = torch.tensor(label_lengths, dtype=torch.long)
    return imgs, labels, label_lengths
