import streamlit as st
from huggingface_hub import hf_hub_download
from TTS.api import TTS
import os

# -----------------------------
# Config: Hugging Face model
# -----------------------------
REPO_ID = "SYSPIN/tts_vits_coquiai_HindiFemale"
MODEL_FILE = "hi_female_vits_30hrs.pt"

# -----------------------------
# Load model with caching
# -----------------------------
@st.cache_resource
def load_model():
    # Download model from Hugging Face hub at runtime
    model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)
    # Load TTS model locally
    tts = TTS(model_path=model_path, progress_bar=False, gpu=False)
    return tts

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Chatterbox TTS - Hindi Female Voice")

tts = load_model()

text = st.text_area("Enter text to synthesize", "Namaste! Aap kaise hain?")

if st.button("Generate Speech"):
    if not text.strip():
        st.warning("Please enter some text to synthesize!")
    else:
        audio_path = "output.wav"
        try:
            tts.tts_to_file(text=text, file_path=audio_path)
            st.success("Audio generated successfully!")
            st.audio(audio_path)
        except Exception as e:
            st.error(f"Error generating speech: {e}")

# Optional: clean up old audio files
if os.path.exists("output.wav"):
    os.remove("output.wav")
