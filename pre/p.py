import os
import re
from pathlib import Path
import pandas as pd
import zipfile
from tqdm import tqdm

# Windows 保留名（不区分大小写）
RESERVED = re.compile(
    r"^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])$",
    re.IGNORECASE,
)


def safe_path_component(name: str) -> str:
    """把 Windows 保留名改成安全名字，例如 Com8 -> _Com8"""
    if RESERVED.match(name):
        return f"_{name}"
    return name


def make_safe_relpath(rel_path: str) -> str:
    """把整条相对路径里的保留名都改掉"""
    parts = Path(rel_path).parts
    safe_parts = [safe_path_component(p) for p in parts]
    return str(Path(*safe_parts))


def extract_only_length4(
    csv_path: str = "./data/dataset.csv",
    zip_path: str = "./data/samples.zip",
    extract_dir: str = "./data/imgs",
):
    """
    只解压 length==4 的图片，并自动避开 Windows 保留目录名
    """
    print("读取 CSV 并筛选 length=4 ...")
    df = pd.read_csv(csv_path)
    df4 = df[df["length"] == 4]
    target_files = set(df4["filepath"].tolist())
    print(f"需要解压的文件数: {len(target_files):,}")

    os.makedirs(extract_dir, exist_ok=True)

    success, missing = 0, 0
    with zipfile.ZipFile(zip_path, "r") as z:
        namelist = set(z.namelist())

        for rel_path in tqdm(target_files, desc="解压 length=4"):
            # 在 zip 里找真实成员名
            candidates = [rel_path, rel_path.lstrip("./")]
            found = next((c for c in candidates if c in namelist), None)
            if found is None:
                missing += 1
                continue

            # 目标本地路径：把保留名改掉
            safe_rel = make_safe_relpath(found)
            target_path = Path(extract_dir) / safe_rel
            target_path.parent.mkdir(parents=True, exist_ok=True)

            # 手动写出文件（比 z.extract 更可控）
            with z.open(found) as src, open(target_path, "wb") as dst:
                dst.write(src.read())
            success += 1

    print(f"\n完成！成功解压 {success:,} 张，未找到 {missing:,} 张")
    print(f"图片目录: {extract_dir}")
    return extract_dir


if __name__ == "__main__":
    extract_only_length4(
        csv_path="./data/dataset.csv",
        zip_path="./data/samples.zip",
        extract_dir="./data/imgs",
    )