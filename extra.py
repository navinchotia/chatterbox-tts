import os
import requests

def download_file(url: str, save_path: str):
    if not os.path.exists(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"Downloading {url} ...")
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"Saved to {save_path}")
    else:
        print(f"File already exists: {save_path}")
    return save_path
