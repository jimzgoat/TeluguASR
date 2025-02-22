from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import torch
import librosa
import os
import subprocess
import shutil
import uuid

# Import ASR Inferencer
from inference import Inferencer

app = FastAPI()

# Serve uploaded audio files
UPLOAD_FOLDER = "/app/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="uploads")

# ASR Model Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
huggingface_folder = "/mnt/d/IIITHCapstone/huggingface-hub45wer"
model_path = "/mnt/d/IIITHCapstone/saved/ASR/checkpoints/ASR/checkpoints45wer/best_model -.45wer.tar"

# Initialize ASR Model
inferencer = Inferencer(device, huggingface_folder, model_path)

@app.get("/")
async def serve_html():
    """Serves the main index.html page."""
    file_path = "/app/index.html"
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return JSONResponse(content={"error": "index.html not found"}, status_code=404)

@app.get("/background.jpg")
async def get_background():
    """Serves the background image."""
    file_path = "/app/background.jpg"
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return JSONResponse(content={"error": "background.jpg not found"}, status_code=404)

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Handles audio transcription:
    - Saves uploaded WebM file
    - Converts it to WAV using FFmpeg
    - Runs ASR transcription using Inferencer
    """
    unique_filename = str(uuid.uuid4()) + ".webm"
    temp_file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
    wav_file_path = temp_file_path.replace(".webm", ".wav")

    try:
        # Save uploaded file
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"✅ Saved WebM File: {temp_file_path}")

        # Convert WebM to WAV using FFmpeg
        # subprocess.run(["ffmpeg", "-i", temp_file_path, "-ar", "16000", wav_file_path], check=True)
        subprocess.run(["ffmpeg", "-i", temp_file_path, "-ac", "1", "-ar", "16000", wav_file_path], check=True)


        print(f"✅ Converted WAV File: {wav_file_path}")

        # Load and process WAV audio
        wav, _ = librosa.load(wav_file_path, sr=16000)
        transcript = inferencer.transcribe(wav)

        response_data = {"transcript": transcript, "filename": unique_filename}
        print(f"🔍 Response Data: {response_data}")  # Debugging Log

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        print(f"📄 WebM file: {temp_file_path}")
        print(f"🎵 WAV file: {wav_file_path}")

    return JSONResponse(content=response_data)

@app.post("/upload-audio")
async def upload_audio(file: UploadFile = File(...)):
    """
    Handles file uploads and saves them to `uploads/`
    """
    output_path = os.path.join(UPLOAD_FOLDER, file.filename)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    with open(output_path, "wb") as f:
        f.write(await file.read())

    return {"message": "File uploaded successfully", "filename": file.filename}
