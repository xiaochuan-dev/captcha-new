import os

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

from config import CHAR2IDX


PARQUET_PATH = "./data/captcha.parquet"

IMAGE_OUTPUT = "./data/captcha_fixed_images.npy"
LABEL_OUTPUT = "./data/captcha_fixed_labels.npy"

IMG_H = 32
IMG_W = 128


# =========================
# 图片预处理
# =========================

def resize_keep_ratio_pad(
    img: Image.Image,
    target_h: int,
    target_w: int,
    fill: int = 255,
):
    src_w, src_h = img.size

    if src_w <= 0 or src_h <= 0:
        return Image.new("L", (target_w, target_h), fill)

    scale = min(
        target_w / src_w,
        target_h / src_h,
    )

    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))

    img = img.resize(
        (new_w, new_h),
        Image.Resampling.BILINEAR,
    )

    canvas = Image.new(
        "L",
        (target_w, target_h),
        fill,
    )

    offset_x = (target_w - new_w) // 2
    offset_y = (target_h - new_h) // 2

    canvas.paste(img, (offset_x, offset_y))

    return canvas


def process_image(
    image_bytes: bytes,
    height: int,
    width: int,
):
    arr = np.frombuffer(
        image_bytes,
        dtype=np.uint8,
    )

    arr = arr.reshape(
        int(height),
        int(width),
        3,
    )

    img = Image.fromarray(
        arr,
        mode="RGB",
    )

    # RGB -> 灰度
    img = img.convert("L")

    # resize + padding
    img = resize_keep_ratio_pad(
        img,
        target_h=IMG_H,
        target_w=IMG_W,
        fill=255,
    )

    # 最终 uint8，范围 0~255
    return np.asarray(
        img,
        dtype=np.uint8,
    )


# =========================
# 主程序
# =========================

def main():
    print(f"读取: {PARQUET_PATH}")

    parquet = pq.ParquetFile(PARQUET_PATH)

    total = parquet.metadata.num_rows

    print(f"样本数: {total:,}")
    print(f"输出图片尺寸: {IMG_H} x {IMG_W}")
    print()

    # ---------------------------------
    # 创建 mmap 文件
    # ---------------------------------

    print("创建图片 mmap...")

    images = np.lib.format.open_memmap(
        IMAGE_OUTPUT,
        mode="w+",
        dtype=np.uint8,
        shape=(total, IMG_H, IMG_W),
    )

    print("创建标签 mmap...")

    labels = np.lib.format.open_memmap(
        LABEL_OUTPUT,
        mode="w+",
        dtype=np.uint8,
        shape=(total, 4),
    )

    # ---------------------------------
    # 逐 row group 处理
    # ---------------------------------

    offset = 0

    for group_idx in tqdm(
        range(parquet.num_row_groups),
        desc="预处理",
    ):
        table = parquet.read_row_group(
            group_idx,
            columns=[
                "image",
                "width",
                "height",
                "label",
            ],
        )

        images_col = table["image"]
        widths_col = table["width"]
        heights_col = table["height"]
        labels_col = table["label"]

        n = table.num_rows

        for i in range(n):

            image_bytes = images_col[i].as_py()
            width = widths_col[i].as_py()
            height = heights_col[i].as_py()
            label = str(labels_col[i].as_py()).lower()

            # -------------------------
            # 图片
            # -------------------------

            image = process_image(
                image_bytes,
                height,
                width,
            )

            images[offset] = image

            # -------------------------
            # label
            # -------------------------

            if len(label) != 4:
                raise ValueError(
                    f"发现非 4 字符标签: "
                    f"index={offset}, label={label!r}"
                )

            for j, char in enumerate(label):
                if char not in CHAR2IDX:
                    raise ValueError(
                        f"未知字符: "
                        f"index={offset}, "
                        f"label={label!r}, "
                        f"char={char!r}"
                    )

                labels[offset, j] = CHAR2IDX[char]

            offset += 1

        # 每个 row group 处理完以后同步一下
        images.flush()
        labels.flush()

    # ---------------------------------
    # 完成
    # ---------------------------------

    images.flush()
    labels.flush()

    del images
    del labels

    print()
    print("=" * 60)
    print("完成！")
    print("=" * 60)

    print(f"图片: {IMAGE_OUTPUT}")
    print(f"标签: {LABEL_OUTPUT}")

    print()
    print("文件大小:")

    print(
        f"images: "
        f"{os.path.getsize(IMAGE_OUTPUT) / 1024**3:.2f} GiB"
    )

    print(
        f"labels: "
        f"{os.path.getsize(LABEL_OUTPUT) / 1024**2:.2f} MiB"
    )


if __name__ == "__main__":
    main()