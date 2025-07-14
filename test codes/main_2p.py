import streamlit as st
from TTS.api import TTS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from kokoro import KPipeline
import soundfile as sf

# --- Page Config ---
st.set_page_config(page_title="🗣️ Multi-Model TTS App", layout="centered")
st.title("🗣️ Multi-Model Text-to-Speech Synthesizer")

# --- Tier and Model Options ---
MODEL_TIERS = {
    "Basic Models": ["Tacotron2", "GlowTTS", "English VITS"],
    "Advanced / Multilingual": ["Coqui TTS", "Kokoro", "FastT5"]
}

BASIC_MODELS = {
    "Tacotron2": "tts_models/en/ljspeech/tacotron2-DDC",
    "GlowTTS": "tts_models/en/ljspeech/glow-tts",
    "English VITS": "tts_models/en/ljspeech/vits",
}

ADVANCED_MODELS = {
    "Coqui TTS": "tts_models/multilingual/multi-dataset/your_tts",
    "Kokoro": "kokoro_82m",
    "FastT5": "fastt5_summarizer"
}

# --- Valid Kokoro Voices (from Hugging Face) ---
kokoro_voice_map = {
    "af_bella": "American Female - Bella",
    "af_nicole": "American Female - Nicole",
    "af_river": "American Female - River",
    "am_adam": "American Male - Adam",
    "am_liam": "American Male - Liam",
    "bf_emma": "British Female - Emma",
    "bm_daniel": "British Male - Daniel",
    "hf_alpha": "Hindi Female - Alpha",
    "hm_omega": "Hindi Male - Omega",
    "jf_alpha": "Japanese Female - Alpha",
    "jm_kumo": "Japanese Male - Kumo",
    "zf_xiaoxiao": "Chinese Female - Xiaoxiao",
    "zm_yunjian": "Chinese Male - Yunjian",
}

# --- Sidebar UI ---
tier = st.sidebar.selectbox("🎚️ Select Tier", list(MODEL_TIERS.keys()))
model_name = st.sidebar.selectbox("🤖 Select Model", MODEL_TIERS[tier])

# Advanced options
if model_name in ["Kokoro", "Coqui TTS"]:
    voice = st.sidebar.selectbox("🎙️ Voice", options=list(kokoro_voice_map.keys()), format_func=lambda v: kokoro_voice_map[v])
    speed = st.sidebar.slider("⏩ Speed", 0.5, 2.0, 1.0, step=0.1)
else:
    voice = None
    speed = 1.0

# --- Input Box ---
text_input = st.text_area("💬 Enter your text below", height=150)

# --- Loaders ---
@st.cache_resource
def load_basic_model(model_key):
    return TTS(model_name=BASIC_MODELS[model_key])

@st.cache_resource
def load_fastt5():
    tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-xsum")
    return tokenizer, model

@st.cache_resource
def load_kokoro(lang_code='a'):
    return KPipeline(lang_code=lang_code)

@st.cache_resource
def load_coqui():
    return TTS(model_name=ADVANCED_MODELS["Coqui TTS"])

# --- Run Button ---
if st.button("🚀 Generate Speech"):
    if not text_input.strip():
        st.warning("⚠️ Please enter some text.")
    else:
        output_path = "output_audio.wav"

        # BASIC MODELS
        if model_name in BASIC_MODELS:
            model = load_basic_model(model_name)
            model.tts_to_file(text=text_input, file_path=output_path)
            st.success(f"✅ Speech generated using {model_name}")
            st.audio(output_path)

        # FASTT5 (summarize + VITS)
        elif model_name == "FastT5":
            tokenizer, model = load_fastt5()
            tokens = tokenizer(text_input, return_tensors="pt", truncation=True, max_length=512)
            summary_ids = model.generate(tokens["input_ids"], max_length=60, num_beams=4)
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            st.subheader("📝 Summarized Text:")
            st.write(summary)

            tts = load_basic_model("English VITS")
            tts.tts_to_file(text=summary, file_path=output_path)
            st.success("✅ Speech from summarized text")
            st.audio(output_path)

        # KOKORO
        elif model_name == "Kokoro":
            lang_code = voice.split("_")[0]
            try:
                kokoro = load_kokoro(lang_code)
                result = kokoro(text_input, voice=voice, speed=speed)
                for i, (_, _, audio) in enumerate(result):
                    sf.write(output_path, audio, 24000)
                    st.success(f"✅ Speech from Kokoro ({kokoro_voice_map[voice]})")
                    st.audio(output_path)
                    break
            except Exception as e:
                st.error(f"❌ Kokoro Error: {str(e)}")

        # COQUI TTS
        elif model_name == "Coqui TTS":
            lang_code = voice.split("_")[0]
            try:
                coqui = load_coqui()
                coqui.tts_to_file(
                    text=text_input,
                    speaker=voice,
                    language=lang_code,
                    file_path=output_path
                )
                st.success(f"✅ Speech from Coqui TTS ({kokoro_voice_map.get(voice, voice)})")
                st.audio(output_path)
            except Exception as e:
                st.error(f"❌ Coqui Error: {str(e)}")
