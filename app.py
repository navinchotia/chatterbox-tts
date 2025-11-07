import streamlit as st
from TTS.utils.synthesizer import Synthesizer
import torch
import os

# ----------------------------
# Settings
# ----------------------------
MODEL_FILE = "hi_female_vits_30hrs.pt"
CONFIG_FILE = "config.json"
OUTPUT_DIR = "outputs"

# Make output folder if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource(show_spinner=True)
def load_model(model_file=MODEL_FILE, config_file=CONFIG_FILE, device="cpu"):
    synthesizer = Synthesizer(
        tts_checkpoint=model_file,
        tts_config=config_file,
        use_cuda=(device != "cpu")
    )
    return synthesizer

device = "cpu"  # Change to "cuda" if GPU is available
tts = load_model(device=device)

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("Hindi Female TTS - VITS")

text_input = st.text_area("Enter text in Hindi or Hinglish", height=150)

if st.button("Generate Speech"):
    if not text_input.strip():
        st.warning("Please enter some text.")
    else:
        output_file = os.path.join(OUTPUT_DIR, "output.wav")
        # Generate speech
        tts.tts_to_file(text=text_input, file_path=output_file)
        st.audio(output_file, format="audio/wav")
        st.success("Speech generated successfully!")
