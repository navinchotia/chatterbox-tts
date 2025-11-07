import streamlit as st
from TTS.api import TTS   # ✅ Correct import
import tempfile
import os

# Streamlit setup
st.set_page_config(page_title="Chatterbox TTS – Hindi Female Voice", page_icon="🪔", layout="centered")
st.title("🪔 Chatterbox TTS – Hindi Female Voice")
st.markdown("Type text in **Hindi or English**, and listen to a natural-sounding **female Hindi voice** 🎙️")

# Load TTS model
@st.cache_resource
def load_tts_model():
    # Coqui multilingual model (includes Hindi) -     return TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
    # Coqui only Hindi l model
    return TTS(model_name="tts_models/hi/mai_female/vits")



tts = load_tts_model()

# Text input
text = st.text_area("Enter your text below:", placeholder="Namaste! Kaise ho aap sab?")

# Generate speech
if st.button("🎧 Generate Voice"):
    if not text.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Generating Hindi female voice..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
                tts.tts_to_file(
                    text=text,
                    file_path=tmpfile.name,
                    language="hi",  # Hindi
                    speaker_wav=None  # default female
                )

                st.audio(tmpfile.name, format="audio/wav")
                st.success("✅ Voice generated successfully!")

                # Download option
                with open(tmpfile.name, "rb") as f:
                    st.download_button(
                        "⬇️ Download Audio",
                        data=f,
                        file_name="hindi_female_voice.wav",
                        mime="audio/wav"
                    )

st.markdown("---")
st.markdown("Created with ❤️ using [Coqui TTS](https://github.com/coqui-ai/TTS) and Streamlit.")
