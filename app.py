import streamlit as st
import torch
import requests
import yaml
from pathlib import Path
from TTS.utils.synthesizer import Synthesizer

# GitHub repo base URL
BASE_URL = "https://raw.githubusercontent.com/utkarsh2299/Fastspeech2_HS/main/hindi/female/model"

# Model and config file names in repo
MODEL_FILE = "model.pth"
CONFIG_FILE = "config.yaml"
ENERGY_STATS = "energy_stats.npz"
FEATS_STATS = "feats_stats.npz"
PITCH_STATS = "pitch_stats.npz"

# Directory to cache downloaded files
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

def download_file(filename):
    url = f"{BASE_URL}/{filename}"
    local_path = CACHE_DIR / filename
    if not local_path.exists():
        r = requests.get(url)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(r.content)
    return local_path

@st.cache_data(show_spinner=True)
def load_model():
    # Download necessary files
    model_path = download_file(MODEL_FILE)
    config_path = download_file(CONFIG_FILE)
    energy_stats_path = download_file(ENERGY_STATS)
    feats_stats_path = download_file(FEATS_STATS)
    pitch_stats_path = download_file(PITCH_STATS)

    # Load config YAML
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Initialize Synthesizer
    synthesizer = Synthesizer(
        tts_checkpoint=model_path,
        tts_config=config_path,
        use_cuda=False,
        energy_stats_path=energy_stats_path,
        pitch_stats_path=pitch_stats_path,
        feats_stats_path=feats_stats_path
    )

    return synthesizer

# Streamlit UI
st.title("Hindi Female TTS Demo (FastSpeech2)")

synthesizer = load_model()

text = st.text_area("Enter text to synthesize:", height=150)

if st.button("Generate Speech") and text.strip():
    st.info("Generating...")
    wav = synthesizer.tts(text)
    out_file = CACHE_DIR / "output.wav"
    synthesizer.save_wav(wav, out_file)
    st.audio(str(out_file), format="audio/wav")
