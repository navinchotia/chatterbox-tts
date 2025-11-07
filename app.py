import os
import torch
import torchaudio
import requests
import streamlit as st

# Import the extra.py utilities from the repo
from extra import TTSTokenizer, VitsCharacters, multilingual_cleaners

# ---------------------- Paths ----------------------
MODEL_URL = "https://huggingface.co/SYSPIN/tts_vits_coquiai_HindiFemale/resolve/main/hi_female_vits_30hrs.pt"
MODEL_PATH = "/tmp/hi_female_vits_30hrs.pt"
CHARS_URL = "https://huggingface.co/SYSPIN/tts_vits_coquiai_HindiFemale/resolve/main/chars.txt"
CHARS_PATH = "/tmp/chars.txt"

# ---------------------- Download files if not exist ----------------------
def download_file(url, local_path):
    if not os.path.exists(local_path):
        st.info(f"Downloading {os.path.basename(local_path)}... ⏳")
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024*1024):  # 1 MB chunks
                f.write(chunk)
        st.success(f"{os.path.basename(local_path)} downloaded!")

# ---------------------- Load TTS model ----------------------
@st.cache_resource
def load_tts_model():
    try:
        download_file(MODEL_URL, MODEL_PATH)
        model = torch.jit.load(MODEL_PATH, map_location="cpu")
        model.eval()
        st.success("✅ TTS model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# ---------------------- Load tokenizer ----------------------
@st.cache_resource
def load_tokenizer():
    try:
        download_file(CHARS_URL, CHARS_PATH)
        with open(CHARS_PATH, "r", encoding="utf-8") as f:
            chars = [line.strip() for line in f if line.strip()]
        # Create character set for VitsCharacters
        characters = VitsCharacters(graphemes="".join(chars))
        tokenizer = TTSTokenizer(text_cleaner=multilingual_cleaners, characters=characters)
        st.success("✅ Tokenizer loaded successfully!")
        return tokenizer
    except Exception as e:
        st.error(f"Error loading tokenizer: {e}")
        return None

# ---------------------- Convert text to tensor ----------------------
def text_to_tensor(tokenizer, text):
    token_ids = tokenizer.text_to_ids(text)
    return torch.LongTensor(token_ids).unsqueeze(0)

# ---------------------- Streamlit UI ----------------------
st.title("🎙 Hindi Female Voice TTS")
st.markdown("Generate natural Hindi speech using a female Hindi voice model.")

tts_model = load_tts_model()
tokenizer = load_tokenizer()

text_input = st.text_area("Enter Hindi text:", "नमस्ते! आप कैसी हैं?", height=120)

if st.button("🔊 Generate Speech"):
    if not tts_model:
        st.error("TTS model not loaded. Please check the download.")
    elif not tokenizer:
        st.error("Tokenizer not loaded. Please check chars.txt file.")
    else:
        try:
            input_tensor = text_to_tensor(tokenizer, text_input)
            with torch.no_grad():
                audio = tts_model(input_tensor)

            # Ensure the audio tensor is 1D
            audio = audio.squeeze().cpu()
            output_path = "/tmp/output.wav"

            # Save as standard PCM WAV
            torchaudio.save(output_path, audio.unsqueeze(0), 22050, encoding="PCM_S", bits_per_sample=16)

            st.success("✅ Speech generated!")
            st.audio(output_path)
        except Exception as e:
            st.error(f"Error generating speech: {e}")
