import os
import streamlit as st
import torch
import torchaudio

# Temporary fix for PyTorch 2.6 "weights_only" issue
old_load = torch.load
def fixed_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return old_load(*args, **kwargs)
torch.load = fixed_load
from TTS.api import TTS
import requests
import json

@st.cache_resource
def load_tts_model():
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "hi_female_vits_30hrs.pt")
    config_path = os.path.join(model_dir, "config.json")

    model_url = "https://huggingface.co/SYSPIN/tts_vits_coquiai_HindiFemale/resolve/main/hi_female_vits_30hrs.pt"

    # Download model if missing
    if not os.path.exists(model_path):
        with st.spinner("📦 Downloading Hindi female TTS model... This will take a few minutes."):
            r = requests.get(model_url)
            r.raise_for_status()
            with open(model_path, "wb") as f:
                f.write(r.content)

    # Create config.json if missing
    if not os.path.exists(config_path):
        st.warning("⚙️ No config found, creating minimal config.json automatically...")
        config = {
            "model": "vits",
            "num_chars": 200,
            "num_speakers": 1,
            "output_sample_rate": 22050,
            "phoneme_language": "hi",
            "run_name": "hi_female_vits_30hrs",
            "use_cuda": False
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    # Load model
    tts = TTS(model_path=model_path, config_path=config_path)
    return tts


# -------------------------- Streamlit UI --------------------------
st.title("🎙 Hindi Female Voice TTS")

tts = load_tts_model()

text_input = st.text_area("Enter Hindi text:", "नमस्ते! आप कैसी हैं?")

if st.button("🎧 Generate Voice"):
    with st.spinner("Generating speech..."):
        wav = tts.tts(text_input)
        output_path = "output.wav"
        torchaudio.save(output_path, torch.tensor([wav]), 22050)
        st.audio(output_path)
        st.success("✅ Speech generated successfully!")
