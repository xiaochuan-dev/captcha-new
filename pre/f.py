import random
import requests
import numpy as np
import pandas as pd
import time
from huggingface_hub import HfApi
from PIL import Image
from io import BytesIO
from gmssl import sm2

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

def download_sichuan_gaokao_item():
    url =  f'https://api.sceea.cn/Handler/ValidateImageHandler.ashx?t={random.random()}'
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        "Referer": "https://cx.sceea.cn/html/GKCJ.htm",
    }

    r = requests.get(url, headers=headers, timeout=10)

    img = Image.open(BytesIO(r.content))
    arr = np.array(img, dtype=np.uint8)
    res = arr.flatten().tolist()
    return res
def download_sichuan_gaokao():
    total = 1000

    image = []

    for i in range(total):
        img = download_sichuan_gaokao_item()
        image.append(img)
        time.sleep(1) 
        print(f'{i} done', flush=True)

    df = pd.DataFrame({
        "image": image
    })

    df.to_parquet("sichuan_gaokao.parquet", engine="pyarrow", compression="zstd")


def sm2_encrypt(data: str) -> str:
    if not data:
        raise ValueError("SM2加密内容不能为空")
    public_key = "9121366953ab694e775b71062461b91b1648316ae32d89ad1b59bc6a4b0a5c6184c9851df7e97b6a4948618c1e7a30d740dca436f0556cadce9bc4d67a179eab"
    sm2_crypt = sm2.CryptSM2(
        public_key=public_key,
        private_key=None,
        mode=1
    )

    cipher_bytes = sm2_crypt.encrypt(data.encode("utf-8"))
    if cipher_bytes is None:
        raise RuntimeError("SM2 加密失败（KDF 结果为 0）")

    return "04" + cipher_bytes.hex()


def generate_plaintext() -> str:
    timestamp_ms = int(time.time() * 1000)
    return f'{{"_t":{timestamp_ms},"_d":{{}}}}'

def download_xinanjiaotong_item():
    plaintext = generate_plaintext()
    ciphertext = sm2_encrypt(plaintext)
    j = ciphertext
    url = f'https://yhxt.swjtu.edu.cn/yethan/code?_j={j}'
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        "Referer": "https://cx.sceea.cn/html/GKCJ.htm",
    }

    r = requests.get(url, headers=headers, timeout=10)
    print(r.json())

if __name__ == '__main__':
    download_xinanjiaotong_item()