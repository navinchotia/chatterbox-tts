import torch
import torchaudio
import streamlit as st
import os
import requests

MODEL_URL = "https://huggingface.co/SYSPIN/tts_vits_coquiai_HindiFemale/resolve/main/hi_female_vits_30hrs.pt"
MODEL_PATH = "hi_female_vits_30hrs.pt"

@st.cache_resource
def load_tts_model():
    # Download once and cache locally
    if not os.path.exists(MODEL_PATH):
        st.info("⬇️ Downloading Hindi female model (~330 MB)… please wait once.")
        with requests.get(MODEL_URL, stream=True) as r:
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    st.info("Loading model…")
    model = torch.jit.load(MODEL_PATH, map_location="cpu")
    model.eval()
    return model

tts_model = load_tts_model()

def synthesize_speech(text, output_path="output.wav"):
    with torch.no_grad():
        audio = tts_model(text)
        if isinstance(audio, tuple):
            audio = audio[0]
        audio = audio.squeeze().cpu()
        torchaudio.save(output_path, audio.unsqueeze(0), 22050)
    return output_path

st.title("🎙 Hindi Female Voice TTS")
text = st.text_area("Enter Hindi text:", "नमस्ते! आप कैसी हैं?")

if st.button("Generate Audio"):
    try:
        output_file = synthesize_speech(text)
        st.audio(output_file)
        st.success("✅ Speech generated successfully!")
    except Exception as e:
        st.error(f"Error generating speech: {e}")
