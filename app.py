import streamlit as st
import numpy as np
from TTS.api import TTS
from scipy.io.wavfile import write
import tempfile
import os
import requests

st.set_page_config(page_title="Hindi TTS", page_icon="🎤")
st.title("Hindi TTS - Female Voice")

# Inputs
text_input = st.text_area("Enter text in Hindi:", "नमस्ते, आप कैसे हैं?")

# GitHub raw URLs for model and config
MODEL_URL = "https://github.com/utkarsh2299/Fastspeech2_HS/raw/main/hindi/female/model/fastspeech2_hindi_female.pth"
CONFIG_URL = "https://github.com/utkarsh2299/Fastspeech2_HS/raw/main/hindi/female/model/config.json"

# Helper to download a file from GitHub
@st.cache_resource(show_spinner=True)
def download_file(url, filename):
    if not os.path.exists(filename):
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return filename

# Download model & config
MODEL_FILE = download_file(MODEL_URL, "fastspeech2_hindi_female.pth")
CONFIG_FILE = download_file(CONFIG_URL, "config.json")

# Load TTS model
@st.cache_resource(show_spinner=True)
def load_tts_model(model_path, config_path):
    return TTS(model_path=model_path, config_path=config_path, progress_bar=False, gpu=False)

tts = load_tts_model(MODEL_FILE, CONFIG_FILE)

# Generate speech
if st.button("Generate Speech"):
    with st.spinner("Generating audio..."):
        wav = tts.tts(text_input)

        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        write(tmp_file.name, 22050, wav)
        tmp_file.close()

        st.audio(tmp_file.name, format="audio/wav")
        st.success("Done! Listen above 🎧")
