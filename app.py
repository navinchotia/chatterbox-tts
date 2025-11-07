import streamlit as st
import torch
from extra import TTSTokenizer, VitsCharacters, multilingual_cleaners

# -------------------------------
# Load TorchScript model
# -------------------------------
@st.cache_resource(show_spinner=True)
def load_model(model_path: str):
    # Load TorchScript
    model = torch.jit.load(model_path, map_location="cpu")
    
    # Load tokenizer/vocab from extra.py
    # Use default config for now; adapt if you have saved config
    characters = VitsCharacters()
    tokenizer = TTSTokenizer(multilingual_cleaners, characters)
    
    return model, tokenizer

# -------------------------------
# Safe text -> IDs
# -------------------------------
def encode_text_safe(text, tokenizer):
    token_ids = []
    for char in text:
        try:
            idx = tokenizer.characters.char_to_id(char)
            token_ids.append(idx)
        except KeyError:
            token_ids.append(tokenizer.blank_id)
    token_ids = tokenizer.intersperse_blank_char(token_ids, use_blank_char=True)
    return torch.LongTensor([token_ids])  # add batch dim

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("Hindi TTS with TorchScript VITS")
st.write("Enter Hindi text:")

text_input = st.text_area("Text", value="नमस्ते, आप कैसे हैं?")

# Replace with path to your downloaded TorchScript model
model_path = "https://huggingface.co/SYSPIN/tts_vits_coquiai_HindiFemale/blob/main/hi_female_vits_30hrs.pt"
model, tokenizer = load_model(model_path)

if st.button("Generate Audio"):
    if not text_input.strip():
        st.warning("Please enter some text!")
    else:
        input_ids = encode_text_safe(text_input, tokenizer)
        try:
            with torch.no_grad():
                # TorchScript forward
                audio = model(input_ids)[0].cpu().numpy()  # adjust depending on model output
            st.audio(audio, format="audio/wav")
        except RuntimeError as e:
            st.error(f"Error generating speech: {e}")
