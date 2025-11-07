import torch
import torchaudio
import streamlit as st
import os

# --------------------------
# Model path and setup
# --------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "huggingface", "hi_female_vits_30hrs.pt")

@st.cache_resource
def load_tts_model():
    st.info("Loading Hindi female TTS model… please wait ⏳")
    model = torch.jit.load(MODEL_PATH, map_location="cpu")
    model.eval()
    return model

tts_model = load_tts_model()

# --------------------------
# TTS generation
# --------------------------
def synthesize_speech(text, output_path="output.wav"):
    # Preprocessing might depend on the model, so this is generic
    with torch.no_grad():
        # Model expects tokenized text (depends on how it was traced)
        # Many Hindi VITS models use phonemes internally, so try plain text first
        audio = tts_model(text)
        if isinstance(audio, tuple):
            audio = audio[0]
        audio = audio.squeeze().cpu()
        torchaudio.save(output_path, audio.unsqueeze(0), 22050)
    return output_path

# --------------------------
# Streamlit UI
# --------------------------
st.title("🎙 Hindi Female Voice TTS")

text = st.text_area("Enter Hindi text:", "नमस्ते! आप कैसी हैं?")
if st.button("Generate Audio"):
    try:
        output_file = synthesize_speech(text)
        st.audio(output_file)
        st.success("✅ Speech generated successfully!")
    except Exception as e:
        st.error(f"Error generating speech: {e}")
