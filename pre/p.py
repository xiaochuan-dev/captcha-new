import zipfile
from huggingface_hub import snapshot_download

def zip_all():
    with zipfile.ZipFile("./data/samples.zip", "r") as z:
        z.extractall("./data/imgs")

def download_all_files():

    snapshot_download("szili2011/captcha-ocr-dataset", 
        repo_type="dataset",
        local_dir="./data",
    )


if __name__ == '__main__':
    zip_all()