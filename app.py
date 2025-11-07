import streamlit as st
from TTS.api import TTS
import soundfile as sf

st.title("Text-to-Speech Demo")

text = st.text_area("Enter text to speak:")
if st.button("Generate Speech"):
    if text.strip():
        tts = TTS("tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)
        wav = tts.tts(text)
        sf.write("output.wav", wav, 22050)
        st.audio("output.wav")
    else:
        st.warning("Please enter some text.")
