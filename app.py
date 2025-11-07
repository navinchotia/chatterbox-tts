import torch
import torchaudio
import streamlit as st
import os
import requests
import streamlit as st


# ---------------------- Paths ----------------------
MODEL_DIR = "hindi_tts_model"
MODEL_URL = "https://huggingface.co/SYSPIN/tts_vits_coquiai_HindiFemale/resolve/main/hi_female_vits_30hrs.pt"
MODEL_PATH = "hi_female_vits_30hrs.pt"
CHARS_PATH = os.path.join(MODEL_DIR, "chars.txt")

# ---------------------- Load TTS model ----------------------
@st.cache_resource
def load_tts_model():
    try:
        st.info("Loading Hindi female TTS model... please wait ⏳")
        model = torch.jit.load(MODE_PATH, map_location="cpu")
        model.eval()
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# ---------------------- Load tokenizer ----------------------
@st.cache_resource
def load_tokenizer():
    try:
        with open(CHARS_PATH, "r", encoding="utf-8") as f:
            chars = [line.strip() for line in f if line.strip()]
        char2idx = {c: i for i, c in enumerate(chars)}
        idx2char = {i: c for i, c in enumerate(chars)}
        return char2idx, idx2char
    except Exception as e:
        st.error(f"Error loading tokenizer: {e}")
        return {}, {}

# ---------------------- Convert text to tensor ----------------------
def text_to_tensor(text, char2idx):
    tokens = []
    for ch in text:
        if ch in char2idx:
            tokens.append(char2idx[ch])
        else:
            tokens.append(char2idx.get("_", 0))  # fallback
    return torch.LongTensor(tokens).unsqueeze(0)

# ---------------------- Streamlit UI ----------------------
st.title("🎙 Hindi Female Voice TTS")
st.markdown("Generate natural Hindi speech using a locally loaded female voice model.")

tts_model = load_tts_model()
char2idx, idx2char = load_tokenizer()

text_input = st.text_area("Enter Hindi text:", "नमस्ते! आप कैसी हैं?", height=120)

if st.button("🔊 Generate Speech"):
    if not tts_model:
        st.error("TTS model not loaded. Please check model path or config.")
    elif not char2idx:
        st.error("Tokenizer not loaded. Please check chars.txt file.")
    else:
        try:
            input_tensor = text_to_tensor(text_input, char2idx)
            with torch.no_grad():
                audio = tts_model(input_tensor)

            # Convert to CPU and save
            audio = audio.squeeze().cpu()
            output_path = "output.wav"
            torchaudio.save(output_path, audio.unsqueeze(0), 22050)
            st.audio(output_path)

        except Exception as e:
            st.error(f"Error generating speech: {e}")
