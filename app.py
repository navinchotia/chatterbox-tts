import streamlit as st
import torch
import torchaudio
import os
from extra import TTSTokenizer, VitsCharacters, multilingual_cleaners

# Path to your model (downloaded from Hugging Face)
MODEL_PATH = "https://huggingface.co/SYSPIN/tts_vits_coquiai_HindiFemale/blob/main/hi_female_vits_30hrs.pt"

# Cache the model to avoid reloading on every interaction
@st.cache_resource
def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model = torch.jit.load(model_path, map_location="cpu")
    model.eval()
    # Initialize tokenizer
    characters = VitsCharacters()
    tokenizer = TTSTokenizer(multilingual_cleaners, characters)
    return model, tokenizer

# Function to generate speech
def synthesize_speech(model, tokenizer, text, output_path="output.wav"):
    # Tokenize the text
    text_ids = tokenizer.text_to_ids(text)
    input_tensor = torch.tensor([text_ids], dtype=torch.int64)
    
    with torch.no_grad():
        # Forward pass through the model
        wav = model(input_tensor)[0]  # Assuming model returns waveform as first element

    # Normalize waveform to [-1, 1]
    wav = wav.squeeze().cpu()
    wav = wav / torch.max(torch.abs(wav))
    
    # Save to wav
    torchaudio.save(output_path, wav.unsqueeze(0), 22050)
    return output_path

# Streamlit UI
st.title("Hindi TTS with Coqui VITS Female Voice")
st.write("Enter Hindi text and generate speech:")

text_input = st.text_area("Enter text here:", value="नमस्ते, आप कैसे हैं?")

if st.button("Generate Speech"):
    try:
        model, tokenizer = load_model(MODEL_PATH)
        output_file = synthesize_speech(model, tokenizer, text_input)
        st.audio(output_file)
        st.success(f"Speech generated and saved as {output_file}")
    except Exception as e:
        st.error(f"Error generating speech: {e}")
