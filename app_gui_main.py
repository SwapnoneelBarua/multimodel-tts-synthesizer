import os
import streamlit as st
from TTS.api import TTS
import soundfile as sf
from pydub import AudioSegment
from io import BytesIO
from transformers import pipeline
from kokoro.pipeline import KPipeline

# Streamlit App Config
st.set_page_config(page_title="🗣️ Multimodel TTS Synthesizer", layout="centered")
st.title("🎙️ Multimodel Text-to-Speech Synthesizer")
st.markdown("🤖 Select a model and customize the voice to synthesize speech from text.")
st.markdown("""
**Developed by [Swapnoneel Barua](https://github.com/SwapnoneelBarua)**  
🔗 [LinkedIn](https://www.linkedin.com/in/swapnoneel-barua/) | [GitHub](https://github.com/SwapnoneelBarua)
""")

# Voice options
kokoro_voices = [
    "af_bella", "af_river", "af_heart", "af_jessica",
    "am_adam", "am_liam", "am_fenrir", "bf_emma",
    "bf_isabella", "bm_george", "em_alex", "em_santa",
    "jf_alpha", "jm_kumo", "zf_xiaoyi"
]

accent_map = {
    "English (US)": "en",
    "English (UK)": "en",
    "Hindi": "hi",
    "Japanese": "ja",
    "French": "fr",
    "German": "de",
    "Spanish": "es",
    "Chinese": "zh",
    "Arabic": "ar"
}

# TTS model loaders
@st.cache_resource
def load_basic_model(name):
    return TTS(name)

@st.cache_resource
def load_kokoro():
    return KPipeline(lang_code='a')

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="t5-small")

# Sidebar
st.sidebar.header("🧠 Select Model Tier")
tier = st.sidebar.radio("Choose a tier", ["Basic", "Multilingual", "Summarizer", "Voice Cloning"])

# Input Text
text_input = st.text_area("📝 Enter your text", placeholder="Type something to synthesize...")

# Tier logic
if tier == "Basic":
    model_label = st.sidebar.selectbox("🧩 Choose a Basic Model", ["Tacotron2", "GlowTTS", "VITS Fast", "FastSpeech2"])

elif tier == "Multilingual":
    model_label = st.sidebar.selectbox("🌍 Choose a Multilingual Model", ["Kokoro", "YourTTS", "VITS Multilingual (🚧 In Progress)"])

    if model_label == "Kokoro":
        speaker = st.sidebar.selectbox("🎙️ Kokoro Voice", kokoro_voices)
        speed = st.sidebar.slider("🎚️ Speed", 0.5, 2.0, 1.0)

    elif model_label == "YourTTS":
        st.warning("🔊 YourTTS requires a sample speaker WAV file.")
        speaker_wav = st.sidebar.file_uploader("Upload a sample speaker .wav file", type=["wav"])
        language = st.sidebar.selectbox("🌍 Choose Language/Accent", list(accent_map.keys()))

elif tier == "Summarizer":
    st.info("📄 Paste a large paragraph and we'll summarize it using T5-small.")

elif tier == "Voice Cloning":
    st.markdown("🎙️ Upload a sample voice and clone it with your own text.")
    st.markdown("🚀 Powered by [F5-TTS (Hugging Face)](https://huggingface.co/spaces/mrfakename/E2-F5-TTS)")
    st.link_button("👉 Open F5-TTS in new tab", "https://huggingface.co/spaces/mrfakename/E2-F5-TTS")

# Synthesize button
synthesize_btn = st.button("🔊 Synthesize")

# Handle synthesis
if synthesize_btn and text_input.strip():
    with st.spinner("Synthesizing..."):
        output_path = "output.wav"
        try:
            # BASIC MODELS
            if tier == "Basic":
                if model_label == "Tacotron2":
                    model = load_basic_model("tts_models/en/ljspeech/tacotron2-DDC")
                elif model_label == "GlowTTS":
                    model = load_basic_model("tts_models/en/ljspeech/glow-tts")
                elif model_label == "VITS Fast":
                    model = load_basic_model("tts_models/en/ljspeech/vits--neon")
                elif model_label == "FastSpeech2":
                    model = load_basic_model("tts_models/en/ljspeech/fastspeech2")
                model.tts_to_file(text_input, file_path=output_path)

            # MULTILINGUAL MODELS
            elif tier == "Multilingual":
                if model_label == "Kokoro":
                    kokoro = load_kokoro()
                    result = kokoro(text_input, voice=speaker, speed=speed)
                    for _, _, audio in result:
                        sf.write(output_path, audio, 24000)

                elif model_label == "YourTTS":
                    if speaker_wav is None:
                        st.error("❌ Please upload a speaker WAV file.")
                        st.stop()
                    lang_code = accent_map.get(language, "en")
                    model = load_basic_model("tts_models/multilingual/multi-dataset/your_tts")
                    model.tts_to_file(text=text_input, speaker_wav=speaker_wav, language=lang_code, file_path=output_path)

                elif model_label == "VITS Multilingual (🚧 In Progress)":
                    st.warning("⚠️ VITS Multilingual support is currently in progress and handled separately.")
                    st.stop()

            # SUMMARIZER
            elif tier == "Summarizer":
                summarizer = load_summarizer()
                summary = summarizer(text_input, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
                st.success("✅ Summary generated:")
                st.markdown(f"> {summary}")
                st.stop()

            # Audio playback and download
            st.success("✅ Speech synthesized successfully!")
            audio_bytes = open(output_path, "rb").read()
            st.audio(audio_bytes, format="audio/wav")

            col1, col2 = st.columns(2)
            with col1:
                st.download_button("⬇️ Download WAV", data=audio_bytes, file_name="output.wav", mime="audio/wav")
            with col2:
                mp3_buffer = BytesIO()
                AudioSegment.from_wav(output_path).export(mp3_buffer, format="mp3")
                st.download_button("⬇️ Download MP3", data=mp3_buffer.getvalue(), file_name="output.mp3", mime="audio/mpeg")

        except Exception as e:
            st.error(f"❌ Error: {e}")
