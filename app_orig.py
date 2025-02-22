from fastapi import FastAPI, UploadFile, File, HTTPException
from inference  import Inferencer
import torch
import librosa
import os

app = FastAPI()

# Initialize the Inferencer
device = "cuda" if torch.cuda.is_available() else "cpu"
huggingface_folder = "/mnt/d/IIITHCapstone/huggingface-hub45wer"  # Update this path
model_path = "/mnt/d/IIITHCapstone/saved/ASR/checkpoints/ASR/checkpoints45wer/best_model -.45wer.tar"  # Update this path if you have a model checkpoint

inferencer = Inferencer(device, huggingface_folder, model_path)

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    if file.content_type not in ["audio/wav", "audio/x-wav", "audio/mpeg"]:
        raise HTTPException(status_code=400, detail="Invalid audio format. Please upload a WAV or MP3 file.")

    # Save the uploaded file temporarily
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as f:
        f.write(await file.read())

    # Load and process the audio file
    try:
        wav, _ = librosa.load(temp_file_path, sr=16000)
        transcript = inferencer.transcribe(wav)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio file: {e}")
    finally:
        os.remove(temp_file_path)  # Clean up the temporary file

    return {"transcript": transcript}
