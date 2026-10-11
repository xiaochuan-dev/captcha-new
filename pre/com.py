import os

from PIL import Image
import pandas as pd
import numpy as np
import json

from .util import ensure_file
from utils import resize_keep_ratio_pad
from config import IMG_H, IMG_W

def write():

    p = ensure_file(
        './data/raw_sichuan_gaokao.parquet',
        'xiaochuan-dev/captcha-new',
        'raw_sichuan_gaokao.parquet'
    )
    
    df = pd.read_parquet(p)
    images = df["image"]

    l = len(images)

    with open('./res.json', 'r') as f:
        datas = json.load(f)


    image = []
    label = []
    
    for i in range(l):
        img = images[i]

        filename = f'./data/imgs/{i:05d}.jpg'

        arr = img.astype(np.uint8).reshape((40, 120))
        img = Image.fromarray(arr, mode='L')
        new_img = resize_keep_ratio_pad(img, IMG_H, IMG_W)
        new_list = np.array(new_img, dtype=np.uint8).flatten().tolist()
        image.append(new_list)
        label.append(datas[filename])
    df = pd.DataFrame({
        "image": image,
        "label": label
    })
    
    df.to_parquet("new_raw_sichuan_gaokao.parquet", engine="pyarrow", compression="zstd")

def com():

    image = []
    label = []

    p1 = ensure_file(
        './data/new_sichuan_gaokao.parquet',
        'xiaochuan-dev/captcha-new',
        'new_sichuan_gaokao.parquet'
    )
    df1 = pd.read_parquet(p1)
    image1 = df1["image"]
    label1 = df1["label"]

    for i in range(len(image1)):
        img = image1[i]
        label_str = label1[i]
        image.append(img)
        label.append(label_str)

    p2 = ensure_file(
        './data/new_raw_sichuan_gaokao.parquet',
        'xiaochuan-dev/captcha-new',
        'new_raw_sichuan_gaokao.parquet'
    )
    df2 = pd.read_parquet(p2)
    image2 = df2["image"]
    label2 = df2["label"]

    for i in range(len(image2)):
        img = image2[i]
        label_str = label2[i]
        image.append(img)
        label.append(label_str)

    df = pd.DataFrame({
        "image": image,
        "label": label
    })
    
    df.to_parquet("final_sichuan_gaokao.parquet", engine="pyarrow", compression="zstd")


if __name__ == '__main__':
    com()