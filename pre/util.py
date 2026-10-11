import numpy as np
import pyarrow.parquet as pq
from PIL import Image

def buffer_to_file(image_bytes: list, output_path: str, height=40, width=120):
    arr = np.frombuffer(
        image_bytes,
        dtype=np.uint8,
    )

    arr = arr.reshape((height, width))

    Image.fromarray(arr, mode='L').save(output_path)
    print(f"图片已保存到: {output_path}")
