import streamlit as st
from TTS.api import TTS

@st.cache_resource
def load_tts_model():
    return TTS(model_name="SYSPIN/tts_vits_coquiai_HindiFemale")

st.title("🎙️ Hindi Female TTS Demo")

tts = load_tts_model()

text = st.text_area("Enter Hindi text:", "नमस्ते, आप कैसे हैं?")

if st.button("Generate Speech"):
    with st.spinner("Generating speech..."):
        output_path = "output.wav"
        tts.tts_to_file(text=text, file_path=output_path)
        st.audio(output_path)
