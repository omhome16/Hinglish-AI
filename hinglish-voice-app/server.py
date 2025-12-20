import os
import time
import io
import logging
import base64
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import ollama
from faster_whisper import WhisperModel
from google.cloud import texttospeech

# --- CONFIGURATION ---
OLLAMA_MODEL = "hinglish-final"  # The name we will use in Ollama
GOOGLE_KEY_PATH = "google_key.json"

# Set Google Credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_KEY_PATH

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HinglishAI")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve Static Files
app.mount("/static", StaticFiles(directory="public"), name="static")

@app.get("/")
async def read_root():
    return FileResponse('public/index.html')

logger.info("🚀 Initializing Hinglish AI Production Server (Ollama Powered)...")

# 1. Load Whisper (Ears)
logger.info("Loading Whisper (Ears)...")
try:
    stt_model = WhisperModel("small", device="cpu", compute_type="int8")
    logger.info("✅ Whisper Loaded")
except Exception as e:
    logger.error(f"❌ Failed to load Whisper: {e}")
    raise e

# 2. Check Ollama (Brain)
logger.info("Checking Ollama Connection...")
try:
    ollama.list() 
    logger.info("✅ Ollama Connected")
except Exception as e:
    logger.error(f"❌ Failed to connect to Ollama. Make sure 'ollama serve' is running! Error: {e}")

# 3. Setup Google TTS (Voice)
logger.info("Connecting to Google TTS (Voice)...")
try:
    tts_client = texttospeech.TextToSpeechClient()
    logger.info("✅ Google TTS Connected")
except Exception as e:
    logger.error(f"❌ Failed to connect to Google TTS: {e}")
    raise e

logger.info("--- 🌟 SYSTEM READY 🌟 ---")

def generate_response(user_text: str) -> str:
    """Generates a Hinglish response using the Ollama local server."""
    try:
        response = ollama.chat(model=OLLAMA_MODEL, messages=[
            {'role': 'user', 'content': user_text},
        ])
        return response['message']['content']
    except Exception as e:
        logger.error(f"Ollama Inference Error: {e}")
        return "Sorry, mere brain mein kuch issue hai abhi. (Brain error)"

def text_to_speech_base64(text: str) -> str:
    """Converts text to speech and returns the base64 encoded MP3."""
    synthesis_input = texttospeech.SynthesisInput(text=text)
    
    # Hindi Neural Voice
    voice_params = texttospeech.VoiceSelectionParams(
        language_code="hi-IN",
        name="hi-IN-Neural2-B" # Male
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=1.10
    )

    response = tts_client.synthesize_speech(
        input=synthesis_input, voice=voice_params, audio_config=audio_config
    )
    
    return base64.b64encode(response.audio_content).decode('utf-8')

@app.post("/chat")
async def chat(text: str = Form(...)):
    """Text-only chat endpoint."""
    logger.info(f"📝 User Text: {text}")
    response_text = generate_response(text)
    logger.info(f"🤖 AI Response: {response_text}")
    return JSONResponse(content={"response": response_text})

@app.post("/tts")
async def tts_endpoint(request: dict = Body(...)):
    """Converts any text to audio (Base64)."""
    text = request.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
        
    try:
        audio_b64 = text_to_speech_base64(text)
        return {"audio": audio_b64}
    except Exception as e:
        logger.error(f"TTS Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/talk")
async def talk(file: UploadFile = File(...)):
    """Voice-to-Voice endpoint. Returns JSON with Text + Audio."""
    start_time = time.time()
    
    # --- STEP 1: HEAR ---
    temp_filename = f"temp_{int(time.time())}.webm"
    try:
        with open(temp_filename, "wb") as f:
            f.write(await file.read())
            
        segments, _ = stt_model.transcribe(temp_filename, language="hi")
        user_text = " ".join([s.text for s in segments]).strip()
        logger.info(f"🗣️ User Said: {user_text}")
        
    except Exception as e:
        logger.error(f"STT Error: {e}")
        return {"error": "Speech recognition failed"}
    finally:
         if os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
            except:
                pass

    if not user_text:
        return {"error": "No speech detected"}

    # --- STEP 2: THINK ---
    ai_text = generate_response(user_text)
    logger.info(f"🤖 AI Said: {ai_text}")

    # --- STEP 3: SPEAK ---
    try:
        audio_b64 = text_to_speech_base64(ai_text)
    except Exception as e:
        logger.error(f"TTS Error: {e}")
        return {"error": "TTS Failed", "user_text": user_text, "ai_text": ai_text}

    total_time = time.time() - start_time
    logger.info(f"⏱️ Latency: {total_time:.2f}s")
    
    return {
        "user_text": user_text,
        "ai_text": ai_text,
        "audio": audio_b64
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)