import os
from .util import ensure_file, buffer_to_file
import pyarrow.parquet as pq
import pandas as pd
import numpy as np

def f():
    p = ensure_file(
        './data/raw_sichuan_gaokao.parquet',
        'xiaochuan-dev/captcha-new',
        'raw_sichuan_gaokao.parquet'
    )

    os.makedirs(
        "data/imgs",
        exist_ok=True,
    )

    df = pd.read_parquet(p)
    images = df["image"]

    l = len(images)

    for i in range(l):
        img = images[i]
        buffer_to_file(image_bytes=img, output_path=f'./data/imgs/{i:05d}.jpg')

        

if __name__ == '__main__':
    f()