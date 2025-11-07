import torch
import torchaudio
import streamlit as st
import os
import requests



import streamlit as st
import torch
import torchaudio
import os

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
        model = torch.jit.load(MODEL_PATH, map_location="cpu")
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
        st.error(f"Error
