import os
import torch
import torchaudio
import requests
import streamlit as st

# ---------------------- Paths & URLs ----------------------
MODEL_URL = "https://huggingface.co/SYSPIN/tts_vits_coquiai_HindiFemale/resolve/main/hi_female_vits_30hrs.pt"
MODEL_PATH = "/tmp/hi_female_vits_30hrs.pt"
CHARS_URL = "https://huggingface.co/SYSPIN/tts_vits_coquiai_HindiFemale/resolve/main/chars.txt"
CHARS_PATH = "/tmp/chars.txt"
SAMPLE_RATE = 22050  # model sample rate

# ---------------------- Download utility ----------------------
def download_file(url, local_path):
    if not os.path.exists(local_path):
        st.info(f"Downloading {os.path.basename(local_path)}... ⏳")
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
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

            # Generate audio (TorchScript model)
            with torch.no_grad():
                output = tts_model(input_tensor)

            # Handle TorchScript output
            if isinstance(output, dict):
                audio_tensor = output.get("wav", None)
                if audio_tensor is None:
                    audio_tensor = list(output.values())[0]  # fallback
            else:
                audio_tensor = output

            # Convert to CPU float tensor
            audio_tensor = audio_tensor.squeeze().cpu().float()

            # Normalize to -1.0..1.0
            audio_tensor = audio_tensor / torch.max(torch.abs(audio_tensor))

            # Save as standard PCM WAV
            output_path = "/tmp/output.wav"
            torchaudio.save(output_path, audio_tensor.unsqueeze(0), SAMPLE_RATE, encoding="PCM_S", bits_per_sample=16)

            st.success("✅ Audio generated successfully!")
            st.audio(output_path)

        except Exception as e:
            st.error(f"Error generating speech: {e}")
