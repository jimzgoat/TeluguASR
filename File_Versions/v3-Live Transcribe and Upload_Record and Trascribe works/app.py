from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import torch
import librosa
import os
import subprocess
import shutil
import uuid
import tempfile

# Import ASR Inferencer
from inference import Inferencer

app = FastAPI()

# Ensure BASE_DIR is correctly set
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ASR Model Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
huggingface_folder = "/mnt/d/IIITHCapstone/huggingface-hub45wer"
model_path = "/mnt/d/IIITHCapstone/saved/ASR/checkpoints/ASR/checkpoints45wer/best_model -.45wer.tar"

inferencer = Inferencer(device, huggingface_folder, model_path)

@app.get("/")
async def serve_html():
    """Serves the main index.html page from the base directory."""
    file_path = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return JSONResponse(content={"error": "index.html not found"}, status_code=404)

@app.get("/background.jpg")
async def get_background():
    """Serves the background image from the base directory."""
    file_path = os.path.join(BASE_DIR, "background.jpg")
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return JSONResponse(content={"error": "background.jpg not found"}, status_code=404)

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Existing transcription endpoint (unchanged).
    """
    allowed_extensions = [".webm", ".wav"]
    file_extension = os.path.splitext(file.filename)[-1].lower()
    if file_extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Only .wav and .webm files are supported.")

    unique_filename = str(uuid.uuid4()) + file_extension
    temp_file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
    wav_file_path = temp_file_path.replace(".webm", ".wav") if file_extension == ".webm" else temp_file_path

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"✅ Saved File: {temp_file_path}")

        if file_extension == ".webm":
            if shutil.which("ffmpeg") is None:
                raise HTTPException(status_code=500, detail="FFmpeg is not installed or not found in the system path.")

            subprocess.run(["ffmpeg", "-i", temp_file_path, "-ac", "1", "-ar", "16000", wav_file_path], check=True)
            print(f"✅ Converted WAV File: {wav_file_path}")

        wav, _ = librosa.load(wav_file_path, sr=16000)
        transcript = inferencer.transcribe(wav)

        response_data = {"transcript": transcript, "filename": file.filename}
        print(f"🔍 Response Data: {response_data}")

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"FFmpeg conversion error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse(content=response_data)

@app.post("/upload-audio")
async def upload_audio(file: UploadFile = File(...)):
    """
    Existing upload endpoint (unchanged).
    """
    output_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(output_path, "wb") as f:
        f.write(await file.read())
    return {"message": "File uploaded successfully", "filename": file.filename}

# ---------------------------------------------------------------------
# NEW ENDPOINT FOR LIVE-CHUNK TRANSCRIPTION
# ---------------------------------------------------------------------
@app.post("/live-transcribe-chunk")
async def live_transcribe_chunk(file: UploadFile = File(...)):
    import tempfile
    allowed_extensions = [".webm", ".wav"]
    file_extension = os.path.splitext(file.filename)[-1].lower()
    if file_extension not in allowed_extensions:
        file_extension = ".webm"

    with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as tmp:
        temp_file_path = tmp.name
        data = await file.read()
        tmp.write(data)

    # If the file is too small, return an empty transcript
    if os.path.getsize(temp_file_path) < 1000:
        os.remove(temp_file_path)
        return JSONResponse(content={"transcript": ""})

    # Convert webm to wav if necessary
    if file_extension == ".webm":
        if shutil.which("ffmpeg") is None:
            raise HTTPException(status_code=500, detail="FFmpeg is not installed")
        wav_file_path = temp_file_path.replace(".webm", ".wav")
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "error",
                "-i", temp_file_path,
                "-vn",
                "-ac", "1",
                "-ar", "16000",
                wav_file_path
            ],
            check=True,
        )
    else:
        wav_file_path = temp_file_path

    # Load and transcribe the audio
    wav, _ = librosa.load(wav_file_path, sr=16000)
    partial_transcript = inferencer.transcribe(wav)

    os.remove(temp_file_path)
    if os.path.exists(wav_file_path):
        os.remove(wav_file_path)

    return JSONResponse(content={"transcript": partial_transcript})
