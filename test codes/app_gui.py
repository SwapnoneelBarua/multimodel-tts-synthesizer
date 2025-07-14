import streamlit as st
from TTS.api import TTS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from kokoro import KPipeline
import soundfile as sf
import torch

# Streamlit Config
st.set_page_config(page_title="🎙️ Multi-TTS Synthesizer", layout="centered")
st.title("🎙️ Universal Text-to-Speech Synthesizer")

# ===================== MODEL CONFIG =====================
MODEL_MAPPING = {
    "Tacotron2": "tts_models/en/ljspeech/tacotron2-DDC",
    "GlowTTS": "tts_models/en/ljspeech/glow-tts",
    "English VITS": "tts_models/en/ljspeech/vits",
    "FastT5 Summary → VITS": "fastt5",
    "Kokoro": "kokoro_82m",
    "Coqui TTS (XTTS-v2)": "tts_models/multilingual/multi-dataset/xtts_v2"
}
BASIC_MODELS = ["Tacotron2", "GlowTTS", "English VITS"]
ADVANCED_MODELS = ["Kokoro", "Coqui TTS (XTTS-v2)"]
SUMMARIZER_MODELS = ["FastT5 Summary → VITS"]

# Sidebar UI
model_label = st.sidebar.selectbox("🧠 Select Model", list(MODEL_MAPPING.keys()))
model_name = MODEL_MAPPING[model_label]
output_path = "output.wav"

# Extra controls for multilingual models
language = None
speaker = None
speed = 1.0
if model_label == "Coqui TTS (XTTS-v2)":
    language = st.sidebar.selectbox("🌍 Language", ["en", "fr", "es", "hi", "pt"])
     # Temporarily load model to fetch speakers
    with st.spinner("Loading Coqui XTTS v2 speaker list..."):
        temp_model = load_coqui_multilingual()
        available_speakers = temp_model.speakers
    speaker = st.sidebar.selectbox("🎙️ Speaker", available_speakers)
elif model_label == "Kokoro":
    voice = st.sidebar.selectbox("🎤 Kokoro Voice", [
        "af_bella", "af_jessica", "af_river", "af_nicole", "af_kore",
        "am_liam", "am_adam", "am_eric", "am_puck", "am_michael"
    ])
    speed = st.sidebar.slider("🐢🦅 Speed", 0.5, 2.0, 1.0, 0.1)

# ===================== MODEL LOADERS =====================
@st.cache_resource
def load_basic_tts(model_name):
    return TTS(model_name=model_name)

@st.cache_resource
def load_coqui_multilingual():
    return TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")

@st.cache_resource
def load_fastt5():
    tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-xsum")
    return tokenizer, model

@st.cache_resource
def load_kokoro():
    return KPipeline(lang_code='a')  # American English

# ===================== MAIN APP =====================
text_input = st.text_area("📝 Enter Text", height=200)

if st.button("🎤 Synthesize Speech"):
    if not text_input.strip():
        st.warning("⚠️ Please enter some text.")
    else:
        try:
            # 🔹 BASIC MODELS
            if model_label in BASIC_MODELS:
                model = load_basic_tts(model_name)
                model.tts_to_file(text=text_input, file_path=output_path)
                st.success(f"✅ Speech generated with {model_label}")
                st.audio(output_path)

            # 🔹 FASTT5 + VITS
            elif model_label == "FastT5 Summary → VITS":
                tokenizer, t5_model = load_fastt5()
                tokens = tokenizer(text_input, return_tensors="pt", truncation=True, max_length=512)
                summary_ids = t5_model.generate(tokens["input_ids"], max_length=60, num_beams=4)
                summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                st.subheader("🧠 Summarized Text")
                st.write(summary)

                # TTS on summary
                model = load_basic_tts("tts_models/en/ljspeech/vits")
                model.tts_to_file(text=summary, file_path=output_path)
                st.success("✅ Speech generated from summary!")
                st.audio(output_path)

            # 🔹 KOKORO MULTILINGUAL
            elif model_label == "Kokoro":
                kokoro = load_kokoro()
                result = kokoro(text_input, voice=voice, speed=speed)
                for i, (_, _, audio) in enumerate(result):
                    sf.write(output_path, audio, 24000)
                    st.success(f"✅ Kokoro voice: {voice}")
                    st.audio(output_path)
                    break

            # 🔹 COQUI MULTILINGUAL (XTTS-v2)
            elif model_label == "Coqui TTS (XTTS-v2)":
                model = load_coqui_multilingual()
                st.write("🧾 Available Coqui Speakers:", model.speakers)
                model.tts_to_file(
                    text=text_input,
                    file_path=output_path,
                    speaker=speaker,
                    language=language
                )
                st.success(f"✅ Coqui XTTS v2: {language} - {speaker}")
                st.audio(output_path)

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")