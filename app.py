import streamlit as st
from TTS.api import TTS
from scipy.io.wavfile import write
import tempfile
import os

st.set_page_config(page_title="Hindi TTS", page_icon="🎤")
st.title("Hindi TTS - Female Voice")

# Inputs
text_input = st.text_area("Enter text in Hindi:", "नमस्ते, आप कैसे हैं?")

# Paths to model and config (relative to repo)
MODEL_PATH = os.path.join("hindi_model", "fastspeech2_hindi_female.pth")
CONFIG_PATH = os.path.join("hindi_model", "config.json")

# Load TTS model
@st.cache_resource(show_spinner=True)
def load_tts_model(model_path, config_path):
    return TTS(model_path=model_path, config_path=config_path, progress_bar=False, gpu=False)

tts = load_tts_model(MODEL_PATH, CONFIG_PATH)

# Generate speech
if st.button("Generate Speech"):
    with st.spinner("Generating audio..."):
        wav = tts.tts(text_input)

        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        write(tmp_file.name, 22050, wav)
        tmp_file.close()

        st.audio(tmp_file.name, format="audio/wav")
        st.success("Done! Listen above 🎧")
