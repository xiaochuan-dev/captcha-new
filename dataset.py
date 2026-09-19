import os
import shutil

import numpy as np
import pyarrow.parquet as pq
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
        print(f"使用本地数据集: {local_path}", flush=True)
        return local_path

    print(f"本地未找到 {local_path}，正在从 Hugging Face 下载...", flush=True)
    print(f"  repo: {repo_id}", flush=True)
    print(f"  file: {filename}", flush=True)

    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)

    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=os.path.dirname(os.path.abspath(local_path)) or ".",
    )

    if os.path.abspath(downloaded) != os.path.abspath(local_path):
        if not os.path.exists(local_path):
            shutil.copy2(downloaded, local_path)
        print(f"已保存到: {local_path}", flush=True)

    return local_path


def bytes_to_rgb(image_bytes: bytes, height: int, width: int) -> np.ndarray:
    """RGB 位图字节 -> (H, W, 3) uint8"""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    return arr.reshape(int(height), int(width), 3)


def resize_keep_ratio_pad(
    img: Image.Image,
    target_h: int = IMG_H,
    target_w: int = IMG_W,
    fill: int = 255,
) -> Image.Image:

    src_w, src_h = img.size

    if src_w <= 0 or src_h <= 0:
        return Image.new(img.mode, (target_w, target_h), fill)

    scale = min(
        target_w / src_w,
        target_h / src_h,
    )

    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))

    img = img.resize(
        (new_w, new_h),
        Image.BILINEAR,
    )

    canvas = Image.new(
        img.mode,
        (target_w, target_h),
        fill,
    )

    offset_x = (target_w - new_w) // 2
    offset_y = (target_h - new_h) // 2

    canvas.paste(
        img,
        (offset_x, offset_y),
    )

    return canvas


def rgb_to_model_input(
    rgb: np.ndarray,
    h: int = IMG_H,
    w: int = IMG_W,
) -> torch.Tensor:

    img = Image.fromarray(
        rgb,
        mode="RGB",
    ).convert("L")

    img = resize_keep_ratio_pad(
        img,
        target_h=h,
        target_w=w,
        fill=255,
    )

    arr = np.asarray(
        img,
        dtype=np.float32,
    ) / 255.0

    return torch.from_numpy(
        arr
    ).unsqueeze(0)


class CaptchaDataset(Dataset):

    def __init__(
        self,
        parquet_path: str | None = None,
        transform=None,
    ):

        path = ensure_parquet(
            parquet_path or LOCAL_PARQUET
        )

        print(
            f"加载数据集: {path}",
            flush=True,
        )

        # 不再使用 pd.read_parquet()
        self.parquet = pq.ParquetFile(path)

        self.transform = transform

        # 总样本数
        self.length = self.parquet.metadata.num_rows

        # 每个 row group 的样本数
        self.group_sizes = [
            self.parquet.metadata.row_group(i).num_rows
            for i in range(self.parquet.num_row_groups)
        ]

        # 每个 row group 的全局起始 index
        self.group_offsets = np.cumsum(
            [0] + self.group_sizes
        )

        # 当前 worker 内缓存
        self._cached_group = None
        self._cached_table = None

        print(
            f"样本数: {self.length:,}",
            flush=True,
        )

        print(
            f"Row groups: {self.parquet.num_row_groups}",
            flush=True,
        )

    def __len__(self):
        return self.length

    def _find_group(self, idx: int):
        """
        根据全局 index 找 row group。
        """
        group = np.searchsorted(
            self.group_offsets,
            idx,
            side="right",
        ) - 1

        local_idx = idx - self.group_offsets[group]

        return int(group), int(local_idx)

    def _load_group(self, group: int):

        # 当前 worker 已经缓存
        if self._cached_group == group:
            return self._cached_table

        # 只读取一个 row group
        table = self.parquet.read_row_group(
            group,
            columns=[
                "image",
                "width",
                "height",
                "label",
            ],
        )

        self._cached_group = group
        self._cached_table = table

        return table

    def __getitem__(self, idx):

        if idx < 0:
            idx += self.length

        if idx < 0 or idx >= self.length:
            raise IndexError(idx)

        group, local_idx = self._find_group(idx)

        table = self._load_group(group)

        image_bytes = table["image"][local_idx].as_py()
        width = table["width"][local_idx].as_py()
        height = table["height"][local_idx].as_py()
        label_str = table["label"][local_idx].as_py()

        rgb = bytes_to_rgb(
            image_bytes,
            height,
            width,
        )

        img = rgb_to_model_input(rgb)

        if self.transform is not None:
            img = self.transform(img)

        label_str = str(label_str).lower()

        indices = [
            CHAR2IDX[c]
            for c in label_str
            if c in CHAR2IDX
        ]

        if not indices:
            indices = [1]

        label = torch.tensor(
            indices,
            dtype=torch.long,
        )

        return img, label, len(label)


def ctc_collate_fn(batch):

    imgs, labels, label_lengths = zip(*batch)

    imgs = torch.stack(
        imgs,
        0,
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