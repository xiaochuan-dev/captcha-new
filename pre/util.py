import os
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from huggingface_hub import hf_hub_download

def ensure_file(
    local_path: str,
    repo_id: str,
    filename: str,
) -> str:
    """
    本地没有则从 Hugging Face 自动下载。
    """
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        print(f"使用本地数据: {local_path}", flush=True)
        return local_path

    print(f"本地未找到 {local_path}", flush=True)
    print(f"正在从 Hugging Face 下载: {filename}", flush=True)
    print(f"  repo: {repo_id}", flush=True)

    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)

    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=os.path.dirname(os.path.abspath(local_path)) or ".",
    )

    if os.path.abspath(downloaded) != os.path.abspath(local_path):
        if not os.path.exists(local_path):
            os.replace(downloaded, local_path)

    print(f"下载完成: {local_path}", flush=True)
    return local_path


def buffer_to_file(image_bytes: list, output_path: str, height=40, width=120):
    arr = image_bytes.astype(np.uint8).reshape((height, width))

    Image.fromarray(arr, mode='L').save(output_path)
    print(f"图片已保存到: {output_path}")
