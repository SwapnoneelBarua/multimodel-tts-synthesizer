import streamlit as st
from TTS.api import TTS
from kokoro import KPipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import soundfile as sf
import os
from pydub import AudioSegment

# ---------- Setup ----------
st.set_page_config(page_title="🎙️ Multi-TTS Synthesizer", layout="centered")
st.title("🎙️ Multi-Model Text-to-Speech Synthesizer")

# ------------------ Credits ------------------
st.markdown("""
**Developed by [Swapnoneel Barua](https://github.com/SwapnoneelBarua)**  
🔗 [LinkedIn](https://www.linkedin.com/in/swapnoneel-barua/) | [GitHub](https://github.com/SwapnoneelBarua)
""")

# ---------- Utility ----------
def convert_wav_to_mp3(wav_path, mp3_path):
    audio = AudioSegment.from_wav(wav_path)
    audio.export(mp3_path, format="mp3")
    return mp3_path

# ---------- Model Loaders ----------
@st.cache_resource
def load_tts_model(model_name):
    return TTS(model_name=model_name)

@st.cache_resource
def load_kokoro(lang='a'):
    return KPipeline(lang_code=lang)

@st.cache_resource
def load_xtts_v2():
    return TTS(model_name="coqui/XTTS-v2")

@st.cache_resource
def load_fastt5():
    tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-xsum")
    return tokenizer, model

# ---------- Sidebar Navigation ----------
tab = st.sidebar.radio("🧭 Navigation", ["Basic Models", "Multilingual / Advanced", "Summarizer"])

# ---------- Input Text ----------
text_input = st.text_area("✍️ Enter Text to Synthesize", height=150)

# ---------- Basic Models ----------
if tab == "Basic Models":
    model_choice = st.selectbox("🛠️ Select Model", ["Tacotron2", "GlowTTS", "VITS"])
    model_map = {
        "Tacotron2": "tts_models/en/ljspeech/tacotron2-DDC",
        "GlowTTS": "tts_models/en/ljspeech/glow-tts",
        "VITS": "tts_models/en/ljspeech/vits"
    }
    model = load_tts_model(model_map[model_choice])

    if st.button("🔊 Synthesize"):
        if not text_input.strip():
            st.warning("Please enter some text.")
        else:
            output_path = "output.wav"
            model.tts_to_file(text=text_input, file_path=output_path)

            st.audio(output_path, format="audio/wav")
            mp3_path = convert_wav_to_mp3(output_path, "output.mp3")

            col1, col2 = st.columns(2)
            with col1:
                with open(output_path, "rb") as f:
                    st.download_button("⬇️ Download WAV", f, file_name="output.wav")
            with col2:
                with open(mp3_path, "rb") as f:
                    st.download_button("⬇️ Download MP3", f, file_name="output.mp3")

# ---------- Multilingual / Advanced ----------
elif tab == "Multilingual / Advanced":
    model_label = st.selectbox("🧠 Choose Model", ["XTTS v2 (Coqui)", "Kokoro"])
    language = st.sidebar.selectbox("🌍 Language", ["en", "fr", "es", "hi", "pt", "de"])

    if model_label == "XTTS v2 (Coqui)":
        with st.spinner("Loading Coqui XTTS v2..."):
            model = load_xtts_v2()
            available_speakers = model.speakers or ['en_0', 'en_1']  # fallback
        speaker = st.selectbox("🎙️ Speaker", available_speakers)

        if st.button("🔊 Synthesize"):
            if not text_input.strip():
                st.warning("Please enter some text.")
            else:
                output_path = "output.wav"
                try:
                    model.tts_to_file(
                        text=text_input,
                        file_path=output_path,
                        speaker=speaker,
                        language=language
                    )
                    st.audio(output_path, format="audio/wav")
                    mp3_path = convert_wav_to_mp3(output_path, "output.mp3")

                    col1, col2 = st.columns(2)
                    with col1:
                        with open(output_path, "rb") as f:
                            st.download_button("⬇️ Download WAV", f, file_name="output.wav")
                    with col2:
                        with open(mp3_path, "rb") as f:
                            st.download_button("⬇️ Download MP3", f, file_name="output.mp3")

                except Exception as e:
                    st.error(f"❌ XTTS Error: {e}")

    elif model_label == "Kokoro":
        voice = st.selectbox("🧬 Voice", [
            "af_bella", "af_river", "af_heart", "af_jessica", "am_adam", "am_liam", "am_fenrir",
            "bf_emma", "bf_isabella", "bm_george", "em_alex", "em_santa", "jf_alpha", "jm_kumo", "zf_xiaoyi"
        ])
        speed = st.slider("⚡ Speed", 0.5, 2.0, 1.0, 0.1)
        lang_map = {"en": "a", "fr": "f", "hi": "h", "es": "e", "jp": "j", "zh": "z"}
        kokoro = load_kokoro(lang_map.get(language, 'a'))

        if st.button("🔊 Synthesize"):
            try:
                result = kokoro(text_input, voice=voice, speed=speed)
                for i, (_, _, audio) in enumerate(result):
                    output_path = f"output.wav"
                    sf.write(output_path, audio, 24000)
                    break

                st.audio(output_path, format="audio/wav")
                mp3_path = convert_wav_to_mp3(output_path, "output.mp3")

                col1, col2 = st.columns(2)
                with col1:
                    with open(output_path, "rb") as f:
                        st.download_button("⬇️ Download WAV", f, file_name="output.wav")
                with col2:
                    with open(mp3_path, "rb") as f:
                        st.download_button("⬇️ Download MP3", f, file_name="output.mp3")

            except Exception as e:
                st.error(f"❌ Kokoro Error: {e}")

# ---------- Summarizer + TTS ----------
elif tab == "Summarizer":
    st.subheader("📄 Summarize & Speak")
    if st.button("🧠 Summarize and Synthesize"):
        if not text_input.strip():
            st.warning("Please enter some text.")
        else:
            tokenizer, model = load_fastt5()
            tokens = tokenizer(text_input, return_tensors="pt", truncation=True, max_length=512)
            summary_ids = model.generate(tokens["input_ids"], max_length=60, num_beams=4)
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            st.success("📝 Summary Generated:")
            st.write(summary)

            tts_model = load_tts_model("tts_models/en/ljspeech/vits")
            output_path = "summary_output.wav"
            tts_model.tts_to_file(text=summary, file_path=output_path)

            st.audio(output_path, format="audio/wav")
            mp3_path = convert_wav_to_mp3(output_path, "summary_output.mp3")

            col1, col2 = st.columns(2)
            with col1:
                with open(output_path, "rb") as f:
                    st.download_button("⬇️ Download WAV", f, file_name="summary_output.wav")
            with col2:
                with open(mp3_path, "rb") as f:
                    st.download_button("⬇️ Download MP3", f, file_name="summary_output.mp3")
