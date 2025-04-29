from fastapi import FastAPI, UploadFile, File
import whisper
import os
import ffmpeg


app = FastAPI()

model = whisper.load_model("base")  

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    
    audio_path = "audio.wav"
    ffmpeg.input(temp_path).output(audio_path, ar=16000, ac=1).run()
    
    
    result = model.transcribe(audio_path)
    
    os.remove(temp_path)
    os.remove(audio_path)
    
    return {"text": result["text"]}
