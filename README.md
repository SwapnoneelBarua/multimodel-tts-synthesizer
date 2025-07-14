# 🗣️ Multimodel TTS Synthesizer

A multilingual, multi-model Text-to-Speech (TTS) synthesizer built with 🐍 Python and 🧠 🐸TTS, Kokoro, YourTTS, and Transformers.

## 🎯 Features

- ✅ **Basic Models:** Tacotron2, GlowTTS, VITS, FastSpeech2
- 🌍 **Multilingual:** Kokoro, YourTTS (speaker cloning)
- 📄 **Summarization:** T5-small (text condensing)
- 🎙️ **Voice Cloning:** via Hugging Face [F5-TTS](https://huggingface.co/spaces/mrfakename/E2-F5-TTS)
- 🎧 **Output:** Download synthesized speech as `.wav` or `.mp3`

---

## 📦 Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/SwapnoneelBarua/multimodel-tts-synthesizer
cd multimodel-tts-synthesizer
```
### 2️⃣ Setup Conda Environment
Create the environment using environment.yml:
```bash
conda env create -f environment.yml
conda activate ttsenv
```
or
manually with requirements.txt:
```bash
conda env create -f environment.yml
conda activate ttsenv
```
or
Create and activate virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```
### ▶️ Run the App
```bash

streamlit run app_gui_main.py
```

Open http://localhost:8501 in your browser.


### 🧠 Model Options & Guide
## 🧩 Basic Models
Tacotron2

GlowTTS

VITS

FastSpeech2

Just select a model and click Synthesize.

### 🌐 Multilingual (Kokoro)
Select voice like af_bella, am_liam, etc.
Adjust speed if needed.

# Note: VITS Multilingual support coming soon.

##📄 Summarizer
Summarize large texts using the T5-small model.
Paste your content, and get a TL;DR.

## 🧬 Voice Cloning
## ➡️ Try F5-TTS for real-time voice cloning

Upload your voice

Enter custom text

Clone the voice output!

### ⚠️ Notes
VITS Multilingual are under integration – stay tuned!

XTTS v2 is disabled due to incompatibility in current 🐸TTS API.

Internet required for loading online models from Hugging Face.

### 🤝 Contribution
Pull requests are welcome. Fork the repo, create a feature branch, and submit a PR.

📄 License
This project is under the MIT License.

👤 Developed By
Swapnoneel Barua
🔗 LinkedIn | GitHub


