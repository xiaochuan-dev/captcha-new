import random
import requests
import numpy as np
import pandas as pd
import time
from huggingface_hub import HfApi
from PIL import Image
from io import BytesIO

def upload(filename):
    api = HfApi()

    REPO_ID = "xiaochuan-dev/captcha-new"
    LOCAL_PARQUET = f"./{filename}"

    api.upload_file(
        path_or_fileobj=LOCAL_PARQUET,
        path_in_repo=filename,
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message="Upload length-4 captcha dataset (RGB bitmap)",
    )

def download_fuzhou_item():
    url = f'https://jwcjwxt2.fzu.edu.cn:82/plus/verifycode.asp?n={random.random()}'
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        "Referer": "https://jwcjwxt2.fzu.edu.cn:82/",
    }

    r = requests.get(url, headers=headers, timeout=10)

    img = Image.open(BytesIO(r.content))
    arr = np.array(img, dtype=np.uint8)
    res = arr.flatten().tolist()
    return res
def download_fuzhou():
    total = 1000

    image = []

    for i in range(total):
        img = download_fuzhou_item()
        image.append(img)
        time.sleep(1) 
        print(f'{i} done', flush=True)

    df = pd.DataFrame({
        "image": image
    })

    df.to_parquet("fuzhou.parquet", engine="pyarrow", compression="zstd")

if __name__ == '__main__':
    download_fuzhou()
    upload('fuzhou.parquet')