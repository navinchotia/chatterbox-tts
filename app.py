import streamlit as st
import torch
import torchaudio as ta
import numpy as np
import tempfile
from chatterbox.tts import ChatterboxTTS
# from chatterbox.mtl_tts import ChatterboxMultilingualTTS  # if multilingual support needed

# --- SETUP PAGE ---
st.set_page_config(
    page_title="Chatterbox TTS Demo",
    page_icon="🗣️",
    layout="centered"
)

st.title("🗣️ Chatterbox TTS Demo")
st.write("Convert text to realistic speech directly in your browser using [Chatterbox TTS](https://github.com/resemble-ai/chatterbox).")

# --- DEVICE SELECTION ---
@st.cache_resource(show_spinner=False)
def load_model(device="cpu"):
    model = ChatterboxTTS.from_pretrained(device=device)
    return model

if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

st.sidebar.header("⚙️ Settings")
st.sidebar.write(f"**Device in use:** `{device}`")

model = load_model(device=device)

# --- USER INPUTS ---
text = st.text_area("Enter the text you want to convert to speech:", height=120,
                    placeholder="Type something like: Hello, this is Chatterbox TTS speaking!")

col1, col2 = st.columns(2)
with col1:
    exaggeration = st.slider("Emotional Exaggeration", 0.0, 1.5, 0.8, 0.1)
with col2:
    cfg_weight = st.slider("CFG Weight (voice adherence)", 0.0, 1.5, 0.8, 0.1)

audio_prompt = st.file_uploader("Optional: Upload a reference voice (WAV format)", type=["wav"], accept_multiple_files=False)

# --- GENERATION BUTTON ---
if st.button("🎧 Generate Speech", use_container_width=True):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Generating speech... Please wait ⏳"):
            gen_kwargs = {
                "exaggeration": exaggeration,
                "cfg_weight": cfg_weight,
            }
            if audio_prompt:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_prompt:
                    tmp_prompt.write(audio_prompt.read())
                    tmp_prompt_path = tmp_prompt.name
                    gen_kwargs["audio_prompt_path"] = tmp_prompt_path

            wav = model.generate(text, **gen_kwargs)

            # Convert to torch tensor (mono, float32)
            if isinstance(wav, np.ndarray):
                wav = torch.from_numpy(wav)
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)
            wav = wav.to(torch.float32)

            # Save to a temporary WAV file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
                ta.save(tmp_wav.name, wav, model.sr)
                audio_bytes = open(tmp_wav.name, "rb").read()

            st.success("✅ Speech generated successfully!")
            st.audio(audio_bytes, format="audio/wav")
            st.download_button("⬇️ Download Audio", data=audio_bytes, file_name="chatterbox_output.wav", mime="audio/wav")
