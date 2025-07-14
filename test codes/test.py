import os
import streamlit as st
from TTS.api import TTS
import soundfile as sf
from pydub import AudioSegment
from io import BytesIO

# App title and layout
st.set_page_config(page_title="🗣️Multimodel TTS Synthesizer", layout="centered")
st.title("🎙️ Multimodel Text-to-Speech Synthesizer")
st.markdown("🤖Select a model and customize the voice to synthesize speech from text.")
st.markdown("""
**Developed by [Swapnoneel Barua](https://github.com/SwapnoneelBarua)**  
🔗 [LinkedIn](https://www.linkedin.com/in/swapnoneel-barua/) | [GitHub](https://github.com/SwapnoneelBarua)
""")

# Define speaker options
xtts_speakers = ["en_0", "en_1"]

# Local XTTS-v2 model paths
xtts_model_dir = "models/xtts_v2"
xtts_config_path = os.path.join(xtts_model_dir, "config.json")
xtts_model_path = os.path.join(xtts_model_dir, "model.pth")
xtts_vocoder_path = os.path.join(xtts_model_dir, "vocoder.pth")

# Load XTTS-v2 model locally
@st.cache_resource
def load_xtts_local():
    return TTS(
        config_path=xtts_config_path,
        model_path=xtts_model_path,
        vocoder_path=xtts_vocoder_path,
        progress_bar=True,
        gpu=False
    )

# Load basic models
@st.cache_resource
def load_basic_model(name):
    return TTS(name)

# Sidebar UI
st.sidebar.header("🧠 Select Model Tier")
tier = st.sidebar.radio("Choose a tier", ["Basic", "Advanced / Multilingual"])

if tier == "Basic":
    model_label = st.sidebar.selectbox("🧩 Choose a basic model", ["Tacotron2", "GlowTTS", "VITS Fast"])

elif tier == "Advanced / Multilingual":
    model_label = st.sidebar.selectbox("🌐 Choose an advanced model", ["XTTS v2 (Local)", "Kokoro"])

# User input
text_input = st.text_area("📝 Enter your text", placeholder="Type something to synthesize...")

# Voice options
if model_label == "XTTS v2 (Local)":
    st.sidebar.markdown("### 🎤 XTTS Voice Settings")
    speaker = st.sidebar.selectbox("🎙️ Choose Speaker", xtts_speakers)
    language = st.sidebar.selectbox("🌍 Language", ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "zh", "ar", "hi", "ja"])

elif model_label == "Kokoro":
    from kokoro.pipeline import KPipeline

    @st.cache_resource
    def load_kokoro():
        return KPipeline(lang_code='a')

    speaker = st.sidebar.selectbox("🎙️ Kokoro Voice", ["af_bella", "af_river", "af_heart", "af_jessica", "am_adam", "am_liam", "am_fenrir",
            "bf_emma", "bf_isabella", "bm_george", "em_alex", "em_santa", "jf_alpha", "jm_kumo", "zf_xiaoyi"])
    language = None
    speed = st.sidebar.slider("🎚️ Speed", 0.5, 2.0, 1.0)

# Synthesize button
synthesize_btn = st.button("🔊 Synthesize")

# Output area
if synthesize_btn and text_input.strip():
    with st.spinner("Synthesizing..."):
        output_path = "output.wav"

        try:
            if tier == "Basic":
                if model_label == "Tacotron2":
                    model = load_basic_model("tts_models/en/ljspeech/tacotron2-DDC")
                elif model_label == "GlowTTS":
                    model = load_basic_model("tts_models/en/ljspeech/glow-tts")
                elif model_label == "VITS Fast":
                    model = load_basic_model("tts_models/en/ljspeech/vits--neon")

                model.tts_to_file(text_input, file_path=output_path)

            elif tier == "Advanced / Multilingual":
                if model_label == "XTTS v2 (Local)":
                    model = load_xtts_local()
                    model.tts_to_file(
                        text=text_input,
                        speaker=speaker,
                        language=language,
                        file_path=output_path
                    )

                elif model_label == "Kokoro":
                    kokoro = load_kokoro()
                    result = kokoro(text_input, voice=speaker, speed=speed)
                    for _, _, audio in result:
                        sf.write(output_path, audio, 24000)

            # Success message and audio playback
            st.success("✅ Speech synthesized successfully!")
            audio_bytes = open(output_path, "rb").read()
            st.audio(audio_bytes, format="audio/wav")

            # Download options
            col1, col2 = st.columns(2)

            with col1:
                st.download_button(
                    label="⬇️ Download WAV",
                    data=audio_bytes,
                    file_name="output.wav",
                    mime="audio/wav"
                )

            with col2:
                # Convert to MP3 using pydub
                audio_segment = AudioSegment.from_wav(output_path)
                mp3_buffer = BytesIO()
                audio_segment.export(mp3_buffer, format="mp3")
                st.download_button(
                    label="⬇️ Download MP3",
                    data=mp3_buffer.getvalue(),
                    file_name="output.mp3",
                    mime="audio/mpeg"
                )

        except Exception as e:
            st.error(f"❌ Error: {e}")