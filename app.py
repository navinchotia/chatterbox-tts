import os
import torch
import torchaudio
import requests
import streamlit as st

# ---------------------- Paths ----------------------
MODEL_URL = "https://huggingface.co/SYSPIN/tts_vits_coquiai_HindiFemale/resolve/main/hi_female_vits_30hrs.pt"
MODEL_PATH = "/tmp/hi_female_vits_30hrs.pt"
CHARS_URL = "https://huggingface.co/SYSPIN/tts_vits_coquiai_HindiFemale/resolve/main/chars.txt"
CHARS_PATH = "/tmp/chars.txt"

# ---------------------- Download files if not exist ----------------------
def download_file(url, local_path):
    if not os.path.exists(local_path):
        st.info(f"Downloading {os.path.basename(local_path)}... ⏳")
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024*1024):  # 1 MB chunks
                f.write(chunk)
        st.success(f"{os.path.basename(local_path)} downloaded!")

# ---------------------- Load TTS model ----------------------
@st.cache_resource
def load_tts_model():
    try:
        download_file(MODEL_URL, MODEL_PATH)
        model = torch.jit.load(MODEL_PATH, map_location="cpu")
        model.eval()
        st.success("✅ TTS model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading m
