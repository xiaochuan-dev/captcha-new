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

    df = pd.read_parquet(p)
    images = np.stack(df["image"].values)

if __name__ == '__main__':
    f()