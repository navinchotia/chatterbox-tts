import streamlit as st
import os
import torch
from TTS.api import TTS

MODEL_REPO = "SYSPIN/tts_vits_coquiai_HindiFemale"
MODEL_URL = "https://huggingface.co/SYSPIN/tts_vits_coquiai_HindiFemale/blob/main/hi_female_vits_30hrs.pt"
MODEL_PATH = os.path.join(MODEL_DIR, "hi_female_vits_30hrs.pt")
CONFIG_PATH = "config.json"

st.set_page_config(page_title="Hindi Female Voice TTS", layout="centered")

@st.cache_resource
def load_tts_model():
    # Ensure model folder exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Download model if not found locally
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading Hindi Female Voice model (~330MB)..."):
            from huggingface_hub import hf_hub_download
            downloaded_file = hf_hub_download(
                repo_id=MODEL_REPO,
                filename="pytorch_model.pt",
                cache_dir=MODEL_DIR
            )
            os.rename(downloaded_file, MODEL_PATH)

    # Load model using Coqui TTS
    tts = TTS(model_path=MODEL_PATH, config_path=CONFIG_PATH, progress_bar=False, gpu=torch.cuda.is_available())
    return tts

st.title("🎙️ Hindi Female Voice TTS")
st.write("Type Hindi or Hinglish text below to generate speech in a natural female Hindi voice.")

# Input text
text_input = st.text_area("Enter text:", placeholder="Namaste! Kaise ho aap?", height=120)

# Button
if st.button("🔊 Generate Speech"):
    if not text_input.strip():
        st.warning("Please enter some text first.")
    else:
        tts = load_tts_model()
        output_path = "output.wav"
        with st.spinner("Generating audio..."):
            tts.tts_to_file(text=text_input, file_path=output_path)
        st.success("✅ Audio generated successfully!")
        st.audio(output_path, format="audio/wav")

st.markdown("---")
st.caption("Powered by Coqui TTS • Model: SYSPIN/tts_vits_coquiai_HindiFemale")
