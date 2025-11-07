import os
import torch
import torchaudio
import requests
import streamlit as st

# ---------------------- Paths & URLs ----------------------
MODEL_URL = "https://huggingface.co/SYSPIN/tts_vits_coquiai_HindiFemale/resolve/main/hi_female_vits_30hrs.pt"
MODEL_PATH = "/tmp/hi_female_vits_30hrs.pt"
CHARS_URL = "https://huggingface.co/SYSPIN/tts_vits_coquiai_HindiFemale/resolve/main/chars.txt"
CHARS_PATH = "/tmp/chars.txt"
SAMPLE_RATE = 22050  # model sample rate

# ---------------------- Download utility ----------------------
def download_file(url, local_path):
    if not os.path.exists(local_path):
        st.info(f"Downloading {os.path.basename(local_path)}... ⏳")
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                f.write(chunk)
        st.success(f"{os.path.basename(local_path)} downloaded!")

# ---------------------- Load TTS model ----------------------
@st.cache_resource
def load_tts_model():
    try:
        download_file(_
