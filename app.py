import os
import torch
import torchaudio
import requests
import streamlit as st

# ---------------------- Paths ----------------------
MODEL_URL = "https://huggingface.co/SYSPIN/tts_vits_coquiai_HindiFemale/resolve/main/hi_female_vits_30hrs.pt"
MODEL_PATH = "/tmp/hi_female_vits_30hrs.pt"  # Download location
CHARS_URL = "https://huggingface.co/SYSPIN/tts_vits_coquiai_HindiFemale/resolve/main/chars.txt"
CHARS_PATH = "/tmp/chars.txt"

# ---------------------- Download files if not exist ----------------------
def download_file(url, local_path):
    if not os.path.exists(local_path):
        st.info(f"Downloading {os.path.basename(local_path)}... ⏳")
        r = requests.get(url, stream=True)
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
        char2idx = {c: i for i, c in enumerate(chars)}
        idx2char = {i: c for i, c in enumerate(chars)}
        st.success("✅ Tokenizer loaded successfully!")
        return char2idx, idx2char
    except Exception as e:
        st.error(f"Error loading tokenizer: {e}")
        return {}, {}

# ---------------------- Convert text to tensor ----------------------
def text_to_tensor(text, char2idx):
    tokens = [char2idx.get(ch, char2idx.get("_", 0)) for ch in text]
    return torch.LongTensor(tokens).unsqueeze(0)

# ---------------------- Streamlit UI ----------------------
st.title("🎙 Hindi Female Voice TTS")
st.markdown("Generate natural Hindi speech using a female Hindi voice model.")

tts_model = load_tts_model()
char2idx, idx2char = load_tokenizer()

text_input = st.text_area("Enter Hindi text:", "नमस्ते! आप कैसी हैं?", height=120)

if st.button("🔊 Generate Speech"):
    if not tts_model:
        st.error("TTS model not loaded. Please check the download.")
    elif not char2idx:
        st.error("Tokenizer not loaded. Please check chars.txt file.")
    else:
        try:
            # Convert text to tensor
            input_tensor = text_to_tensor(text_input, char2idx)

            # Generate audio using the model's forward/infer method
            with torch.no_grad():
                # If your model is TorchScript, sometimes the method is 'forward'
                # and sometimes 'infer', check the model source
                if hasattr(tts_model, "infer"):
                    audio = tts_model.infer(input_tensor)
                else:
                    audio = tts_model(input_tensor)

            # Ensure audio is 1D float tensor
            audio = audio.squeeze().cpu()
            if audio.ndim != 1:
                audio = audio.mean(dim=0)

            # Save as standard WAV without TorchCodec
            output_path = "/tmp/output.wav"
            torchaudio.save(output_path, audio.unsqueeze(0), 22050, encoding="PCM_S", bits_per_sample=16)

            st.audio(output_path)

        except Exception as e:
            st.error(f"Error generating speech: {e}")
