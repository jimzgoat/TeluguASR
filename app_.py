from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from inference import Inferencer
import torch
import librosa
import os

app = FastAPI()

# Initialize the Inferencer
device = "cuda" if torch.cuda.is_available() else "cpu"
huggingface_folder = "/mnt/d/IIITHCapstone/huggingface-hub45wer"  # Update this path
model_path = "/mnt/d/IIITHCapstone/saved/ASR/checkpoints/ASR/checkpoints45wer/best_model -.45wer.tar"  # Update this path if you have a model checkpoint

inferencer = Inferencer(device, huggingface_folder, model_path)

import subprocess

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    print("Received file:", file.filename)
    temp_file_path = f"temp_{file.filename}"
    wav_file_path = f"temp_{file.filename.split('.')[0]}.wav"

    try:
        with open(temp_file_path, "wb") as f:
            f.write(await file.read())
        print("File saved temporarily.")

        # Convert webm to wav using ffmpeg
        subprocess.run(["ffmpeg", "-i", temp_file_path, "-ar", "16000", wav_file_path], check=True)
        print("File converted to WAV.")

        # Process the WAV file
        wav, _ = librosa.load(wav_file_path, sr=16000)
        transcript = inferencer.transcribe(wav)
        print("Generated transcription:", transcript)
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Error processing audio file.")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        if os.path.exists(wav_file_path):
            os.remove(wav_file_path)

    return {"transcript": transcript}


@app.get("/", response_class=HTMLResponse)
async def serve_html():
    html_content = """
    <!DOCTYPE html>
    <html lang=\"en\">
    <head>
        <meta charset=\"UTF-8\">
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
        <title>Voice Recorder</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f4f4f9;
                color: #333;
                margin: 0;
                padding: 0;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                height: 100vh;
            }

            h1 {
                color: #4CAF50;
            }

            button {
                margin: 10px;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
                cursor: pointer;
            }

            button:disabled {
                background-color: #ccc;
                cursor: not-allowed;
            }

            audio {
                margin-top: 20px;
            }

            #transcriptionResult {
                margin-top: 20px;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 5px;
                background-color: #fff;
                width: 80%;
                max-width: 600px;
                text-align: center;
                font-size: 16px;
            }
        </style>
    </head>
    <body>
        <h1>Voice Recorder</h1>
        <button id=\"startButton\">Start Recording</button>
        <button id=\"stopButton\" disabled>Stop Recording</button>
        <audio id=\"audioPlayback\" controls></audio>
        <br>
        <button id=\"uploadButton\" disabled>Upload and Transcribe</button>
        <p id=\"transcriptionResult\"></p>

        <script>
            let mediaRecorder;
            let audioChunks = [];
            const startButton = document.getElementById('startButton');
            const stopButton = document.getElementById('stopButton');
            const uploadButton = document.getElementById('uploadButton');
            const audioPlayback = document.getElementById('audioPlayback');
            const transcriptionResult = document.getElementById('transcriptionResult');

            startButton.addEventListener('click', async () => {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream, {
                    mimeType: 'audio/webm',
                    audioBitsPerSecond: 16000 // Adjusting the bit rate to ensure compatibility
                });

                mediaRecorder.ondataavailable = event => {
                    if (event.data.size > 0) {
                        audioChunks.push(event.data);
                    }
                };

                mediaRecorder.onstop = () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                    const audioUrl = URL.createObjectURL(audioBlob);
                    audioPlayback.src = audioUrl;
                    uploadButton.disabled = false;
                };

                audioChunks = [];
                mediaRecorder.start();
                startButton.disabled = true;
                stopButton.disabled = false;
            });

            stopButton.addEventListener('click', () => {
                mediaRecorder.stop();
                startButton.disabled = false;
                stopButton.disabled = true;
            });

            uploadButton.addEventListener('click', async () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                const formData = new FormData();
                formData.append('file', audioBlob, 'recording.webm');

                const response = await fetch('/transcribe/', {
                    method: 'POST',
                    body: formData
                });

                if (response.ok) {
                    const result = await response.json();
                    transcriptionResult.textContent = `Transcription: ${result.transcript}`;
                } else {
                    transcriptionResult.textContent = 'Failed to transcribe audio.';
                }
            });
        </script>
    </body>
    </html>
    """
    return html_content

@app.post("/upload-audio")
async def upload_audio(file: UploadFile = File(...)):
    output_path = f"uploads/{file.filename}"
    os.makedirs("uploads", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(await file.read())
    return {"message": "File uploaded successfully", "filename": file.filename}