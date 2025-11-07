import streamlit as st
import torch
import torchaudio
import requests
from io import BytesIO
from extra import TTSTokenizer, VitsCharacters, multilingual_cleaners

# Hugging Face model URL
MODEL_URL = "https://huggingface.co/SYSPIN/tts_vits_coquiai_HindiFemale/resolve/main/hi_female_vits_30hrs.pt"

# Cache the model to avoid reloading
@st.cache_resource
def load_model(model_url):
    # Download model into memory
    response = requests.get(model_url)
    if response.status_code != 200:
        raise FileNotFoundError(f"Could not download model from {model_url}")
    
    buffer = BytesIO(response.content)
    model = torch.jit.load(buffer, map_location="cpu")
    model.eval()
    
    # Initialize tokenizer
    characters = VitsCharacters()
    tokenizer = TTSTokenizer(multilingual_cleaners, characters)
    return model, tokenizer

# Function to synthesize speech
def synthesize_speech(model, tokenizer, text):
    text_ids = tokenizer.text_to_ids(text)
    input_tensor = torch.tensor([text_ids], dtype=torch.int64)
    
    with torch.no_grad():
        wav = model(input_tensor)[0]
    
    wav = wav.squeeze().cpu()
    wav = wav / torch.max(torch.abs(wav))
    return wav

# Streamlit UI
st.title("Hindi TTS (Coqui VITS Female Voice)")
text_input = st.text_area("Enter Hindi text here:", value="नमस्ते, आप कैसे हैं?")

if st.button("Generate Speech"):
    try:
        model, tokenizer = load_model(MODEL_URL)
        wav = synthesize_speech(model, tokenizer, text_input)
        
        # Save to BytesIO for streaming directly
        buffer = BytesIO()
        torchaudio.save(buffer, wav.unsqueeze(0), 22050, format="wav")
        st.audio(buffer.getvalue(), format="audio/wav")
        st.success("Speech generated successfully!")
    except Exception as e:
        st.error(f"Error generating speech: {e}")
