import streamlit as st
from chatterbox_tts import TTS
import tempfile
import os

# Initialize app
st.set_page_config(page_title="Chatterbox TTS – Hindi Female Voice", page_icon="🪔", layout="centered")
st.title("🪔 Chatterbox TTS – Hindi Female Voice")
st.markdown("Type text in **Hindi or English**, and listen to a natural-sounding **female Hindi voice** 🎙️")

# Initialize TTS model (Coqui XTTS v2 multilingual)
@st.cache_resource
def load_tts_model():
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    return tts

tts = load_tts_model()

# Text input
text = st.text_area("Enter your text below:", placeholder="Namaste! Kaise ho aap sab?")

# Voice and language configuration
language = "hi"  # Hindi
speaker_wav = None  # Use model's default female Hindi voice

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
                    speaker_wav=speaker_wav,
                    language=language
                )
                st.audio(tmpfile.name, format="audio/wav")
                st.success("✅ Voice generated successfully!")

                # Offer download
                with open(tmpfile.name, "rb") as file:
                    st.download_button(
                        label="⬇️ Download Audio",
                        data=file,
                        file_name="hindi_female_voice.wav",
                        mime="audio/wav"
                    )

# Footer
st.markdown("---")
st.markdown("Created with ❤️ using [Coqui TTS](https://github.com/coqui-ai/TTS) and Streamlit.")
