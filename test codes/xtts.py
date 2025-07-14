import streamlit as st
from TTS.api import TTS
import tempfile
import os

# Title of the app
st.set_page_config(page_title="Multilingual XTTS Voice Synthesizer", layout="centered")
st.title("🗣️ XTTS v2 - Multilingual Voice Synthesizer (Remote Model)")

# Upload target voice
speaker_wav = st.file_uploader("🎙️ Upload a voice sample (WAV only)", type=["wav"])

# Input text
text_input = st.text_area("💬 Enter the text to synthesize", height=100)

# Select language
language = st.selectbox("🌐 Select language", ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "ko", "hi"])

# Output path
output_path = os.path.join("output", "xtts_output.wav")
os.makedirs("output", exist_ok=True)

# Load model from 🐸TTS hub
@st.cache_resource
def load_xtts_remote():
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    tts.to("cuda" if tts.is_cuda else "cpu")
    return tts

tts = load_xtts_remote()

# Synthesize button
if st.button("🔊 Synthesize"):
    if not text_input:
        st.warning("Please enter some text.")
    elif not speaker_wav:
        st.warning("Please upload a voice sample (.wav).")
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
            tmp_wav.write(speaker_wav.read())
            speaker_path = tmp_wav.name

        with st.spinner("Synthesizing audio..."):
            tts.tts_to_file(
                text=text_input,
                speaker_wav=speaker_path,
                file_path=output_path,
                language=language
            )

        st.success("✅ Synthesis complete!")
        st.audio(output_path, format="audio/wav")
        st.download_button("⬇️ Download", output_path, file_name="xtts_output.wav")
