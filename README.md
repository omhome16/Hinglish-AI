# 🗣️ Hinglish Voice AI

**A production-ready, voice-enabled AI assistant that speaks Hinglish (Hindi-English) using a fine-tuned LLaMA 3.2 model.**

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![LLaMA](https://img.shields.io/badge/LLaMA-3.2--3B-orange?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green?style=flat-square&logo=fastapi)
![HuggingFace](https://img.shields.io/badge/🤗%20Model-omhome%2Fhinglish--llama--r8-yellow?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## 🎯 Overview

Hinglish Voice AI is a complete speech-to-speech conversational AI system that understands and responds in **Hinglish** (a natural blend of Hindi and English commonly spoken in India). The project demonstrates end-to-end ML engineering—from dataset creation and model fine-tuning to deployment with a modern web interface.

### ✨ Key Features

- **🎙️ Voice-to-Voice Interaction** — Speak naturally and get spoken responses in Hinglish
- **🧠 Fine-tuned LLaMA 3.2** — Custom QLoRA fine-tuning on Hinglish conversational dataset
- **👂 Whisper STT** — Real-time speech recognition using Faster-Whisper
- **🔊 Neural TTS** — Natural-sounding responses via Google Cloud Text-to-Speech
- **🌐 Modern Web UI** — Glassmorphism design with reactive audio visualizations
- **⚡ Local Inference** — Runs on consumer hardware via Ollama (no cloud LLM costs)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐ │
│  │  Voice Orb  │    │  Text Chat  │    │  Audio Visualizer   │ │
│  │  (Hold to   │    │  (Type or   │    │  (Web Audio API)    │ │
│  │   speak)    │    │   listen)   │    │                     │ │
│  └─────────────┘    └─────────────┘    └─────────────────────┘ │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Backend                             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  /talk  →  Voice-to-Voice (Full Pipeline)               │   │
│  │  /chat  →  Text-only conversation                       │   │
│  │  /tts   →  Text-to-Speech conversion                    │   │
│  └─────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────────┐
│  Faster       │   │  Ollama +     │   │  Google Cloud     │
│  Whisper      │   │  Fine-tuned   │   │  Text-to-Speech   │
│  (STT)        │   │  LLaMA 3.2    │   │  (Neural Voice)   │
│               │   │  (Hinglish)   │   │                   │
└───────────────┘   └───────────────┘   └───────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **LLM** | LLaMA 3.2 3B (QLoRA fine-tuned, Q4_K_M quantized) |
| **Fine-tuning** | Unsloth + TRL + PEFT |
| **Inference Runtime** | Ollama |
| **Speech-to-Text** | Faster-Whisper (Small model) |
| **Text-to-Speech** | Google Cloud Neural Voice (hi-IN) |
| **Backend** | FastAPI + Uvicorn |
| **Frontend** | Vanilla JS + Tailwind CSS + Web Audio API |

---

## 📁 Project Structure

```
hinglish-ai/
├── data/
│   ├── blended_dataset/        # Training data
│   ├── eval_data/              # Evaluation results
│   └── scripts/                # Data processing scripts
├── hinglish-voice-app/
│   ├── server.py               # FastAPI backend
│   ├── Modelfile               # Ollama model configuration
│   └── public/
│       └── index.html          # Web interface
├── train.py                    # QLoRA fine-tuning script
├── judge.py                    # LLM-as-Judge evaluation
└── requirements.txt            # Python dependencies
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running
- Google Cloud account with TTS API enabled

### Installation

```bash
# Clone the repository
git clone https://github.com/omhom16/hinglish-ai.git
cd hinglish-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download the model from Hugging Face (example)
# Place hinglish-llama-3.2-3b-q4_k_m.gguf in hinglish-voice-app/

# Create Ollama model
cd hinglish-voice-app
ollama create hinglish-final -f Modelfile

# Set up Google Cloud credentials
# Place your google_key.json in hinglish-voice-app/
```

### Running the Application

```bash
# Start Ollama server (in a separate terminal)
ollama serve

# Start the application
cd hinglish-voice-app
python server.py
```

Open `http://localhost:8000` in your browser.

---

## 🎓 Model Training

The model was fine-tuned using **QLoRA** (Quantized Low-Rank Adaptation) for efficient training:

```bash
python train.py \
    --r 16 \
    --alpha 32 \
    --output_dir ./checkpoints \
    --dataset omhome/hinglish-blended-dataset \
    --repo omhome/hinglish-llama-r8
```

### 🤗 Hugging Face Resources

| Resource | Link |
|----------|------|
| **Fine-tuned Model** | [omhome/hinglish-llama-r8](https://huggingface.co/omhome/hinglish-llama-r8) |
| **Training Dataset** | [omhome/hinglish-blended-dataset](https://huggingface.co/datasets/omhome/hinglish-blended-dataset) |

### Training Details

| Parameter | Value |
|-----------|-------|
| Base Model | `meta-llama/Llama-3.2-3b-instruct` |
| LoRA Rank (r) | 16 |
| LoRA Alpha | 32 |
| Learning Rate | 2e-4 |
| Epochs | 1 |
| Batch Size | 4 (with gradient accumulation) |
| Quantization | 4-bit NF4 |

---

## 📊 Evaluation

Model quality was assessed using **LLM-as-Judge** methodology with Gemini as the evaluator:

```bash
python judge.py
```

The evaluation compares base LLaMA responses against fine-tuned responses on criteria:
- **Natural Hinglish** — Authenticity of code-mixing
- **Helpfulness** — Quality of the answer
- **Technical Accuracy** — Correctness for technical queries

---

## 🎨 User Interface

The web interface features:

- **Interactive Orb** — Hold to record, releases to get response
- **Real-time Visualization** — Audio-reactive animations via Web Audio API
- **Text Chat** — Type messages with Listen button for TTS
- **Responsive Design** — Mobile-first with collapsible chat panel

---

## 📄 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/talk` | POST | Voice-to-voice (audio file → text + audio response) |
| `/chat` | POST | Text-only chat (returns JSON response) |
| `/tts` | POST | Text-to-speech (returns base64 audio) |

---

## 👨‍💻 Author

**Om Nawathe**

- GitHub: [@omhom16](https://github.com/omhom16)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

