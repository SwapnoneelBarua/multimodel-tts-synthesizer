import streamlit as st
import torch
from TTS.api import TTS as CoquiTTS
from kokoro import KPipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import soundfile as sf
import os

# App config
st.set_page_config(page_title="🎙️ Multi-Tier TTS Synthesizer", layout="centered")
st.title("🎙️ Multi-Tier Text-to-Speech Synthesizer")

# Tier setup
TIER_OPTIONS = {
    "Basic": ["Tacotron2", "GlowTTS", "English VITS"],
    "Advanced": ["Coqui TTS", "Kokoro", "FastT5"],
    "Multilingual": ["Coqui TTS", "Kokoro"]
}

# Voice Options for advanced/multilingual
VOICE_OPTIONS = {
    "Kokoro": ["af_bella", "af_adam", "af_ken", "uk_emily", "jp_sakura"],
    "Coqui TTS": ["en", "hi", "fr", "de"]
}

# Sidebar controls
tier = st.sidebar.selectbox("🔽 Select Tier", list(TIER_OPTIONS.keys()))
model_name = st.sidebar.selectbox("🎛️ Select Model", TIER_OPTIONS[tier])

# Customization for advanced/multilingual
voice = None
speed = 1.0

if model_name in VOICE_OPTIONS:
    voice = st.sidebar.selectbox("🗣️ Select Voice", VOICE_OPTIONS[model_name])
    speed = st.sidebar.slider("🎚️ Speech Speed", 0.5, 2.0, 1.0, 0.1)

# Text input
text_input = st.text_area("📝 Enter text to synthesize:", height=150)

# Model loading functions
@st.cache_resource
def load_basic_tts(name):
    models = {
        "Tacotron2": "tts_models/en/ljspeech/tacotron2-DDC",
        "GlowTTS": "tts_models/en/ljspeech/glow-tts",
        "English VITS": "tts_models/en/ljspeech/vits"
    }
    return CoquiTTS(model_name=models[name])

@st.cache_resource
def load_fastt5():
    tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-xsum")
    return tokenizer, model

@st.cache_resource
def load_kokoro():
    return KPipeline(lang_code='m')

@st.cache_resource
def load_coqui_multilingual():
    return CoquiTTS(model_name="tts_models/multilingual/multi-dataset/your_tts")  # Replace with actual multilingual model name if needed

# Synthesize button
if st.button("▶️ Synthesize"):
    if not text_input.strip():
        st.warning("⚠️ Please enter some text.")
    else:
        output_path = "output.wav"

        # BASIC TTS
        if model_name in ["Tacotron2", "GlowTTS", "English VITS"]:
            model = load_basic_tts(model_name)
            model.tts_to_file(text_input, file_path=output_path)

        # FASTT5 SUMMARY + SPEECH
        elif model_name == "FastT5":
            tokenizer, model = load_fastt5()
            inputs = tokenizer(text_input, return_tensors="pt", truncation=True, max_length=512)
            summary_ids = model.generate(inputs["input_ids"], max_length=60, num_beams=4)
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            st.subheader("📝 Summarized Text:")
            st.write(summary)
            coqui = load_basic_tts("English VITS")
            coqui.tts_to_file(text=summary, file_path=output_path)

        # KOKORO MULTILINGUAL
        elif model_name == "Kokoro":
            kokoro = load_kokoro()
            result = kokoro(text_input, voice=voice, speed=speed)
            for i, (_, _, audio) in enumerate(result):
                sf.write(output_path, audio, 24000)
                break

        # COQUI MULTILINGUAL
        elif model_name == "Coqui TTS":
            model = load_coqui_multilingual()
            model.tts_to_file(text_input, file_path=output_path, speaker=voice)

        # Output
        st.success("✅ Speech synthesized successfully!")
        st.audio(output_path)

        # Optional: Download button
        with open(output_path, "rb") as f:
            st.download_button("⬇️ Download Audio", f, file_name="output.wav")
