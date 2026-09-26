import os
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from config import IMG_H, IMG_W, HF_REPO  # 如果还需要自动下载的话

# =========================
# 路径配置
# =========================
LOCAL_IMAGES = "./data/images.npy"
LOCAL_LABELS = "./data/labels.npy"
OUTPUT_PARQUET = "./data/captcha.parquet"

# 可选：分批写入，避免一次性把所有数据塞进内存
BATCH_SIZE = 10_000


def convert_npy_to_parquet(
    images_path: str = LOCAL_IMAGES,
    labels_path: str = LOCAL_LABELS,
    output_path: str = OUTPUT_PARQUET,
    batch_size: int = BATCH_SIZE,
):
    print(f"加载图片: {images_path}")
    images = np.load(images_path, mmap_mode="r")  # (N, H, W) uint8

    print(f"加载标签: {labels_path}")
    labels = np.load(labels_path, mmap_mode="r")  # (N, 4) uint8

    # 基本检查
    assert images.ndim == 3
    assert images.shape[1] == IMG_H and images.shape[2] == IMG_W
    assert images.dtype == np.uint8

    assert labels.ndim == 2 and labels.shape[1] == 4
    assert labels.dtype == np.uint8
    assert len(images) == len(labels)

    n = len(images)
    print(f"总样本数: {n:,}")
    print(f"图片 shape: {images.shape}, dtype: {images.dtype}")
    print(f"标签 shape: {labels.shape}, dtype: {labels.dtype}")

    # Parquet schema
    # image: 原始位图 bytes（H*W 个 uint8）
    # labels: list<uint8> 长度固定为 4
    schema = pa.schema([
        ("image", pa.binary()),
        ("labels", pa.list_(pa.uint8(), list_size=4)),  # fixed-size list
        # 也可以再加一些元数据列（可选）
        # ("height", pa.int16()),
        # ("width", pa.int16()),
    ])

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    writer = pq.ParquetWriter(
        output_path,
        schema=schema,
        compression="zstd",          # 对 raw uint8 效果通常比 snappy 好
        # compression_level=3,       # 可选
        use_dictionary=False,        # binary 列一般不需要字典
    )

    try:
        for start in tqdm(range(0, n, batch_size), desc="写入 Parquet"):
            end = min(start + batch_size, n)

            # 取出当前 batch
            img_batch = images[start:end]      # (B, H, W)
            lab_batch = labels[start:end]      # (B, 4)

            # 把每张图转成 bytes
            # 注意：必须用 .tobytes() 或 .ravel().tobytes()
            image_bytes = [
                img.tobytes() for img in img_batch
            ]

            # labels 转成 list of list
            label_lists = lab_batch.tolist()

            batch = pa.RecordBatch.from_arrays(
                [
                    pa.array(image_bytes, type=pa.binary()),
                    pa.array(label_lists, type=pa.list_(pa.uint8(), list_size=4)),
                ],
                schema=schema,
            )

            writer.write_batch(batch)

    finally:
        writer.close()

    print(f"\n转换完成 → {output_path}")
    print(f"文件大小: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    convert_npy_to_parquet()