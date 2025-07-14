import streamlit as st
from TTS.api import TTS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from kokoro import KPipeline
import soundfile as sf
import os

# App setup
st.set_page_config(page_title="🗣️ Multi-Model TTS Synthesizer", layout="centered")
st.title("🗣️ Text-to-Speech + Summarizer GUI")

# Model tiers and mapping
TIER_OPTIONS = {
    "Basic Models": ["Tacotron2", "GlowTTS", "English VITS"],
    "Advanced Models": ["Coqui TTS", "Kokoro"],
    "Summarizer": ["FastT5"]
}

MODEL_MAPPING = {
    "Tacotron2": "tts_models/en/ljspeech/tacotron2-DDC",
    "GlowTTS": "tts_models/en/ljspeech/glow-tts",
    "English VITS": "tts_models/en/ljspeech/vits",
    "Coqui TTS": "tts_models/multilingual/multi-dataset/your_model_name_here",  # Replace with valid
    "FastT5": "google/pegasus-xsum",
    "Kokoro": "kokoro_82m"
}

kokoro_voice_map = {
    "af_bella": "Bella (American English)",
    "af_nicole": "Nicole (American English)",
    "am_eric": "Eric (American Male)",
    "am_fenrir": "Fenrir (American Male)",
    "af_kore": "Kore (American Female)",
    "am_liam": "Liam (American Male)"
}

# Sidebar selections
selected_tier = st.sidebar.radio("🧠 Select Model Tier", list(TIER_OPTIONS.keys()))
model_name = st.sidebar.selectbox("🎯 Choose a Model", TIER_OPTIONS[selected_tier])

voice = None
speed = 1.0

if model_name == "Kokoro":
    voice = st.sidebar.selectbox("🎙️ Voice", options=list(kokoro_voice_map.keys()),
                                  format_func=lambda v: kokoro_voice_map[v])
    speed = st.sidebar.slider("🐢🔁 Speed", 0.5, 2.0, 1.0, 0.1)

# Input area
text_input = st.text_area("💬 Enter your text here:", height=200)

# Loaders
@st.cache_resource
def load_tts_model(model_id):
    return TTS(model_name=model_id)

@st.cache_resource
def load_kokoro(lang_code='a'):
    return KPipeline(lang_code=lang_code)

@st.cache_resource
def load_fastt5():
    tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-xsum")
    return tokenizer, model

# Execution
if st.button("🚀 Synthesize"):
    if not text_input.strip():
        st.warning("⚠️ Please enter some text.")
    else:
        output_path = "output.wav"
        try:
            # BASIC MODELS
            if model_name in ["Tacotron2", "GlowTTS", "English VITS"]:
                model = load_tts_model(MODEL_MAPPING[model_name])
                model.tts_to_file(text=text_input, file_path=output_path)

            # FASTT5 SUMMARIZER
            elif model_name == "FastT5":
                tokenizer, fmodel = load_fastt5()
                inputs = tokenizer(text_input, return_tensors="pt", truncation=True, max_length=512)
                summary_ids = fmodel.generate(inputs["input_ids"], max_length=60, num_beams=4)
                summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                st.subheader("📝 Summary")
                st.write(summary)

                tts_model = load_tts_model("tts_models/en/ljspeech/vits")
                tts_model.tts_to_file(text=summary, file_path=output_path)

            # KOKORO ADVANCED
            elif model_name == "Kokoro":
                lang_code = voice[0]  # Only first letter
                kokoro = load_kokoro(lang_code)
                result = kokoro(text_input, voice=voice, speed=speed)
                for i, (_, _, audio) in enumerate(result):
                    sf.write(output_path, audio, 24000)
                    break

            # COQUI ADVANCED
            elif model_name == "Coqui TTS":
                model = load_tts_model(MODEL_MAPPING[model_name])
                model.tts_to_file(text=text_input, file_path=output_path)  # Assume default speaker/lang

            st.success("✅ Speech synthesized successfully!")
            st.audio(output_path)

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
