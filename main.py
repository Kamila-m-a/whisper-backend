from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import whisper
import torch
import numpy as np
import os
import warnings

# Suppress harmless warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")
warnings.filterwarnings("ignore", category=UserWarning, module="whisper")

# Initialize NumPy explicitly (critical for Whisper)
np.zeros(1)  # Prevents "Numpy is not available" errors

app = FastAPI()

# Enable CORS (required for Flutter)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

# Optimized model loading for Render free tier
model = whisper.load_model("tiny.en", device="cpu")
model.eval()  # Disable dropout for inference
torch.set_num_threads(1)  # Limit CPU threads

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    temp_path = "temp_audio.wav"
    try:
        # 1. Save uploaded file
        with open(temp_path, "wb") as f:
            f.write(await file.read())
        
        # 2. Process with memory limits
        with torch.inference_mode():
            result = model.transcribe(temp_path, fp16=False)  # Force FP32
        
        return {"text": result["text"]}
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {str(e)}"
        )
    finally:
        # 3. Cleanup temp files
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/")
def health_check():
    return {
        "status": "Ready",
        "model": "tiny.en",
        "endpoints": {
            "transcribe": "POST /transcribe",
            "docs": "GET /docs"
        }
    }
