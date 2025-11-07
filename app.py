import streamlit as st
import torch
import json
import requests
from scipy.io.wavfile import write
from io import BytesIO
from pathlib import Path

# -------------------
# Helper functions
# -------------------

@st.cache_data
def download_file(url, local_path):
    """Download a file from URL if it does not exist locally"""
    local_path = Path(local_path)
    if not local_path.exists():
        st.info(f"Downloading {url}...")
        r = requests.get(url)
        r.raise_for_status()
        local_path.write_bytes(r.content)
    return str(local_path)

@st.cache_data
def load_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

@st.cache_resource
def load_model(model_url, config_url, device="cpu"):
    # Download files
    MODEL_FILE = download_file(model_url, "fastspeech2_hindi_female.pt")
    CONFIG_FILE = download_file(config_url, "config.json")
    
    config = load_config(CONFIG_FILE)
    
    # Dynamic imports from model repo
    import sys
    sys.path.append("model")  # assuming code expects this structure
    from model import model as model_module
    from model import utils as utils_module
    from model import vocoder as vocoder_module

    model = model_module.FastSpeech2(config)
    checkpoint = torch.load(MODEL_FILE, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    processor = utils_module.TextProcessor(config)
    
    return model, processor, vocoder_module

def synthesize(model, processor, vocoder_module, text, device="cpu"):
    tokens = processor.text_to_sequence(text)
    tokens = torch.LongTensor(tokens).unsqueeze(0).to(device)
    
    with torch.no_grad():
        mel_outputs, _, _ = model(tokens)
    
    # Convert mel spectrogram to waveform
    audio = vocoder_module.vocoder_infer(mel_outputs)
    return audio

# -------------------
# Streamlit UI
# -------------------

st.title("Hindi Female TTS - FastSpeech2")
st.write("Type your Hindi text below (in Hindi script or transliteration):")
input_text = st.text_area("Text", "नमस्ते, आप कैसे हैं?")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Direct URLs from GitHub raw content
MODEL_URL = "https://raw.githubusercontent.com/utkarsh2299/Fastspeech2_HS/main/hindi/female/model/fastspeech2_hindi_female.pt"
CONFIG_URL = "https://raw.githubusercontent.com/utkarsh2299/Fastspeech2_HS/main/hindi/female/model/config.json"

# Load model
model, processor, vocoder_module = load_model(MODEL_URL, CONFIG_URL, device=device)

if st.button("Generate Speech"):
    audio = synthesize(model, processor, vocoder_module, input_text, device=device)
    
    # Save temporary WAV
    output_path = "output.wav"
    write(output_path, 22050, audio)  # adjust sample rate if needed
    st.audio(output_path, format="audio/wav")
    st.success("Speech generated!")
