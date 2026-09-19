import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

RESERVED = re.compile(
    r"^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])$",
    re.IGNORECASE,
)


def safe_path_component(name: str) -> str:
    if RESERVED.match(name):
        return f"_{name}"
    return name


def make_safe_relpath(rel_path: str) -> str:
    parts = Path(rel_path).parts
    return str(Path(*[safe_path_component(p) for p in parts]))


def create_length4_parquet(
    csv_path: str = "./data/dataset.csv",
    img_root: str = "./data/imgs",
    output_path: str = "./data/captcha_length4.parquet",
    batch_size: int = 2000,
):
    """
    把 length==4 的图片写成 parquet，存储 RGB 位图。

    每行字段：
        - image:   uint8 位图字节 (H*W*3)
        - width:   宽
        - height:  高
        - label:   4 字符标签
        - filepath: 原始路径
    """
    print("读取 dataset.csv ...")
    df = pd.read_csv(csv_path)
    df4 = df[df["length"] == 4].reset_index(drop=True)
    print(f"共找到 {len(df4):,} 张 length=4 的图片")

    schema = pa.schema([
        ("image", pa.binary()),
        ("width", pa.int32()),
        ("height", pa.int32()),
        ("label", pa.string()),
        ("filepath", pa.string()),
    ])

    writer = None
    img_root = Path(img_root)
    success, fail = 0, 0

    for start in tqdm(range(0, len(df4), batch_size), desc="写入 parquet"):
        batch = df4.iloc[start : start + batch_size]

        images, widths, heights, labels, filepaths = [], [], [], [], []

        for _, row in batch.iterrows():
            original_rel = row["filepath"]
            safe_rel = make_safe_relpath(original_rel)
            full_path = img_root / safe_rel

            if not full_path.exists():
                full_path = img_root / original_rel
            if not full_path.exists():
                fail += 1
                continue

            try:
                with Image.open(full_path) as im:
                    im = im.convert("RGB")                # 保持彩色
                    arr = np.asarray(im, dtype=np.uint8)  # shape: (H, W, 3)

                h, w, _ = arr.shape
                images.append(arr.tobytes())
                widths.append(w)
                heights.append(h)
                labels.append(row["text"])
                filepaths.append(original_rel)
                success += 1

            except Exception:
                fail += 1
                continue

        if not images:
            continue

        table = pa.Table.from_pydict(
            {
                "image": images,
                "width": widths,
                "height": heights,
                "label": labels,
                "filepath": filepaths,
            },
            schema=schema,
        )

        if writer is None:
            writer = pq.ParquetWriter(
                output_path,
                schema=schema,
                compression="zstd",
            )
        writer.write_table(table)

    if writer is not None:
        writer.close()

    print(f"\n完成！成功 {success:,} 张，失败 {fail:,} 张")
    print(f"已保存到: {output_path}")
    return output_path


if __name__ == "__main__":
    create_length4_parquet(
        csv_path="./data/dataset.csv",
        img_root="./data/imgs",
        output_path="./data/captcha_length4.parquet",
        batch_size=2000,
    )