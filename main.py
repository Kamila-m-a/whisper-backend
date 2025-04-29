from fastapi import FastAPI, UploadFile, File
import whisper
import os
import ffmpeg
import uvicorn  
import numpy as np

def load_model_safely():
    try:
        # Test NumPy first
        np.array([1, 2, 3])  # Simple check to ensure NumPy works
        return whisper.load_model("tiny") 
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")
        
app = FastAPI()

model = load_model_safely()

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    try :
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        audio_path = "audio.wav"
        ffmpeg.input(temp_path).output(audio_path, ar=16000, ac=1).run()
    
    
        result = model.transcribe(audio_path)

        return {"text": result["text"]}

    finally :    
        if os.remove(temp_path):
            os.remove(temp_path)

        if os.path.exists(audio_path):
            os.remove(audio_path)
      

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000) 
