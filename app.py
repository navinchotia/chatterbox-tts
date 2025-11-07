import streamlit as st
import requests
from TTS.api import TTS
import os

# ---------------------------
# Download model from Hugging Face
# ---------------------------
MODEL_URL = "https://huggingface.co/SYSPIN/tts_vits_coquiai_HindiFemale/resolve/main/hi_female_vits_30hrs.pt"
MODEL_FILE = "hi_female_vits_30hrs.pt"

if not os.path.exists(MODEL_FILE):
    with st.spinner("Downloading TTS model..."):
        r = requests.get(MODEL_URL, stream=True)
        total_length = r.headers.get('content-length')

        with open(MODEL_FILE, "wb") as f:
            if total_length is None:
                f.write(r.content)
            else:
                dl = 0
                total_length = int(total_length)
                for data in r.iter_content(chunk_size=4096):
                    dl += len(data)
                    f.write(data)
                    st.progress(min(dl / total_length, 1.0))

# ---------------------------
# Load the model
# ---------------------------
@st.cache_resource
def load_model(model_path):
    tts = TTS(model_path, progress_bar=False, gpu=False)
    return tts

tts = load_model(MODEL_FILE)

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Hindi TTS with Coqui VITS")
text_input = st.text_area("Enter text to synthesize:", "")

if st.button("Generate Audio"):
    if text_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Generating speech..."):
            audio_path = "output.wav"
            tts.tts_to_file(text=text_input, file_path=audio_path)
            st.audio(audio_path, format="audio/wav")
            st.success("Done!")
