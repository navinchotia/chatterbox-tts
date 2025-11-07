import os
import streamlit as st
import torch
import torchaudio
from TTS.api import TTS

# -----------------------------------------------------------------------------
# Step 1: Define paths and download model automatically if missing
# -----------------------------------------------------------------------------
@st.cache_resource
def load_tts_model():
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "hi_female_vits_30hrs.pt")
    config_path = os.path.join(model_dir, "config.json")

    # Hugging Face direct file URLs (from SYSPIN/tts_vits_coquiai_HindiFemale)
    model_url = "https://huggingface.co/SYSPIN/tts_vits_coquiai_HindiFemale/blob/main/hi_female_vits_30hrs.pt"
   

    # Download the model if not already present
    if not os.path.exists(model_path):
        import requests
        with st.spinner("Downloading Hindi female TTS model... (may take 2–3 min)"):
            r = requests.get(model_url)
            r.raise_for_status()
            with open(model_path, "wb") as f:
                f.write(r.content)

    # Download config file if not present
    if not os.path.exists(config_path):
        import requests
        with st.spinner("Downloading configuration..."):
            r = requests.get(config_url)
            r.raise_for_status()
            with open(config_path, "wb") as f:
                f.write(r.content)

    # Load the model
    tts = TTS(model_path=model_path, config_path=config_path)
    return tts

# -----------------------------------------------------------------------------
# Step 2: Streamlit UI
# -----------------------------------------------------------------------------
st.title("🎙️ Hindi Female Voice TTS")

tts = load_tts_model()

text_input = st.text_area("Enter Hindi text:", "नमस्ते! आप कैसी हैं?")

if st.button("Generate Voice"):
    with st.spinner("Generating speech..."):
        wav = tts.tts(text_input)
        output_path = "output.wav"
        torchaudio.save(output_path, torch.tensor([wav]), 22050)
        st.audio(output_path)
        st.success("Done! ✅")
