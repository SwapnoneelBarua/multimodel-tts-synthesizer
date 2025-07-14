import streamlit as st
import os
from TTS.api import TTS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import tempfile
import torch
import soundfile as sf

# Title
st.set_page_config(page_title="🗣️ Multi-Model TTS Synthesizer", layout="centered")
st.title("🎙️ Multi-Model TTS Synthesizer GUI")

# Accent options
accent_options = ["English - US", "English - UK", "Hindi", "Japanese", "Korean", "French", "German", "Spanish", "Chinese", "Arabic", "Russian", "Portuguese"]
kokoro_voices = ["af_bella", "af_river", "af_heart", "af_jessica", "am_adam", "am_liam", "am_fenrir",
                 "bf_emma", "bf_isabella", "bm_george", "em_alex", "em_santa", "jf_alpha", "jm_kumo", "zf_xiaoyi"]

# TTS loader cache
@st.cache_resource
def load_tts_model(model_name, use_cuda=True):
    tts = TTS(model_name)
    if use_cuda:
        tts.to("cuda")
    return tts

# Summarizer cache
@st.cache_resource
def load_summarizer():
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    return tokenizer, model

# Output audio controls
def audio_controls(output_path):
    audio_bytes = open(output_path, 'rb').read()
    st.audio(audio_bytes, format='audio/wav')
    st.download_button("⬇️ Download Audio", audio_bytes, file_name="output.wav")

# Section selector
option = st.sidebar.selectbox(
    "🔘 Choose a Section",
    ("Basic TTS Models", "Multilingual Models", "Summarizer (FastT5)", "Voice Cloning")
)

# TEXT input
text_input = st.text_area("📝 Enter Text to Synthesize", height=150)

# ========== 1. BASIC TTS ==========
if option == "Basic TTS Models":
    st.subheader("🚀 Fast Basic Models")

    basic_model = st.selectbox("Select Model", ["Tacotron2", "GlowTTS", "VITS", "FastSpeech2"])
    if st.button("🔊 Synthesize"):
        model_name = f"tts_models/en/ljspeech/{basic_model.lower()}"
        tts = load_tts_model(model_name)
        output_path = "basic_output.wav"
        tts.tts_to_file(text=text_input, file_path=output_path)
        audio_controls(output_path)

# ========== 2. MULTILINGUAL TTS ==========
elif option == "Multilingual Models":
    st.subheader("🌐 Multilingual TTS")
    multilingual_model = st.selectbox("Select Model", ["Kokoro", "VITS Multilingual", "YourTTS"])

    if multilingual_model == "Kokoro":
        voice = st.selectbox("🎙️ Select Kokoro Voice", kokoro_voices)
        model_name = "kokoro-ai/kokoro-tts"
        if st.button("🔊 Synthesize"):
            tts = load_tts_model(model_name)
            output_path = "kokoro_output.wav"
            tts.tts_to_file(text=text_input, speaker=voice, file_path=output_path)
            audio_controls(output_path)

    else:
        accent = st.selectbox("🌍 Select Accent/Language", accent_options)
        model_name = "tts_models/multilingual/multi-dataset/your_tts" if multilingual_model == "YourTTS" else "tts_models/multilingual/multi-dataset/vits"
        if st.button("🔊 Synthesize"):
            tts = load_tts_model(model_name)
            output_path = "multi_output.wav"
            tts.tts_to_file(text=text_input, file_path=output_path, language="en")  # Optional: Map accent to code
            audio_controls(output_path)

# ========== 3. SUMMARIZER ==========
elif option == "Summarizer (FastT5)":
    st.subheader("🧠 Summarize then Speak (FastT5)")
    if st.button("📝 Summarize + 🔊 Synthesize"):
        tokenizer, model = load_summarizer()
        inputs = tokenizer("summarize: " + text_input, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        st.success(f"📝 Summary: {summary}")

        tts = load_tts_model("tts_models/en/ljspeech/glow-tts")
        output_path = "summary_output.wav"
        tts.tts_to_file(text=summary, file_path=output_path)
        audio_controls(output_path)

# ========== 4. VOICE CLONING ==========
elif option == "Voice Cloning":
    st.subheader("🧬 Voice Cloning Mode")
    speaker_wav = st.file_uploader("🎤 Upload a speaker sample (.wav)", type=["wav"])
    if speaker_wav:
        speaker_wav_path = os.path.join(tempfile.gettempdir(), "speaker.wav")
        with open(speaker_wav_path, "wb") as f:
            f.write(speaker_wav.getbuffer())

    if st.button("🔊 Clone Voice and Synthesize"):
        if not speaker_wav:
            st.error("Please upload a speaker WAV file.")
        else:
            tts = load_tts_model("tts_models/multilingual/multi-dataset/xtts_v2")
            output_path = "clone_output.wav"
            tts.tts_to_file(text=text_input, file_path=output_path, speaker_wav=speaker_wav_path, language="en")
            audio_controls(output_path)

# Footer
st.markdown("---")
st.caption("Made with ❤️ using [🐸TTS](https://github.com/coqui-ai/TTS) and Streamlit")
