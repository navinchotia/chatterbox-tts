import streamlit as st
from TTS.api import TTS
import os
import requests

@st.cache_resource
def load_tts_model():
    return TTS(model_name="SYSPIN/tts_vits_coquiai_HindiFemale")

st.title("🎙️ Hindi Female TTS Demo")

MODEL_DIR = "hindi_tts_model"
MODEL_URL = "https://huggingface.co/SYSPIN/tts_vits_coquiai_HindiFemale/blob/main/hi_female_vits_30hrs.pt"
MODEL_PATH = os.path.join(MODEL_DIR, "hi_female_vits_30hrs.pt")

os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading Hindi Female TTS model..."):
        r = requests.get(MODEL_URL)
        open(MODEL_PATH, "wb").write(r.content)

tts = load_tts_model()

text = st.text_area("Enter Hindi text:", "नमस्ते, आप कैसे हैं?")

if st.button("Generate Speech"):
    with st.spinner("Generating speech..."):
        output_path = "output.wav"
        tts.tts_to_file(text=text, file_path=output_path)
        st.audio(output_path)
