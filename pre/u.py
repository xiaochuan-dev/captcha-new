import random
import requests
import numpy as np
import pandas as pd
import time
import base64
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
if __name__ == '__main__':
    upload('raw1_sichuan_gaokao.parquet')