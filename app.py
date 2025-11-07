import streamlit as st
import torch
from TTS.api import TTS
from extra import TTSTokenizer, VitsCharacters, multilingual_cleaners

# -------------------------------
# Helper: Safe text to IDs
# -------------------------------
def encode_text_safe(text, tokenizer):
    """
    Encode text into token IDs, replacing unknown chars with blank.
    """
    token_ids = []
    for char in text:
        try:
            idx = tokenizer.characters.char_to_id(char)
            token_ids.append(idx)
        except KeyError:
            # replace unknown char with blank
            token_ids.append(tokenizer.blank_id)
    # intersperse blank char as VITS expects
    token_ids = tokenizer.intersperse_blank_char(token_ids, use_blank_char=True)
    return torch.LongTensor([token_ids])  # batch dimension

# -------------------------------
# Load VITS model
# -------------------------------
@st.cache_resource(show_spinner=True)
def load_model(model_name: str):
    tts = TTS(model_name=model_name, progress_bar=False, gpu=False)
    
    # Load vocab from model config
    config = tts.tts_config
    characters, _ = VitsCharacters.init_from_config(config)
    tokenizer = TTSTokenizer(multilingual_cleaners, characters)
    
    return tts, tokenizer

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("Hindi TTS with VITS")
st.write("Enter Hindi text below and generate speech:")

text_input = st.text_area("Text", value="नमस्ते, आप कैसे हैं?")

model_name = "SYSPIN/tts_vits_coquiai_HindiFemale"  # HF repo
tts, tokenizer = load_model(model_name)

if st.button("Generate Audio"):
    if not text_input.strip():
        st.warning("Please enter some text!")
    else:
        # Safe encoding
        input_ids = encode_text_safe(text_input, tokenizer)
        
        # Generate audio
        try:
            wav = tts.tts_with_torchscript(input_ids=input_ids, speaker=None, language=None)
            st.audio(wav, format="audio/wav")
        except RuntimeError as e:
            st.error(f"Error generating speech: {e}")
