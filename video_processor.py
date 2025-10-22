import os
import whisper
import tempfile
import subprocess
from typing import Optional
import torch

MODEL = None

def get_model():
    """Lazy load the Whisper model."""
    global MODEL
    if MODEL is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        MODEL = whisper.load_model("base", device=device)
    return MODEL

def extract_audio(video_path: str, audio_path: Optional[str] = None) -> str:
    """Extract audio to 16kHz mono WAV using ffmpeg CLI. Returns path to a temp WAV file."""
    if audio_path is None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        audio_path = tmp.name
        tmp.close()
    cmd = [
        "ffmpeg",
        "-y",  # overwrite if exists
        "-i", video_path,
        "-vn",  # no video
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        audio_path,
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed (code {proc.returncode}): {proc.stderr.decode(errors='ignore')}")
        return audio_path
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found on PATH. Please install ffmpeg and ensure it's accessible.")

def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio file using Whisper and return transcript as string."""
    model = get_model()
    use_fp16 = torch.cuda.is_available()
    result = model.transcribe(audio_path, fp16=use_fp16)
    text = result["text"]
    if isinstance(text, list):
        text = " ".join(map(str, text))
    return str(text)

def process_video(video_path: str) -> str:
    """Extract audio, transcribe it, and return transcript as string."""
    audio_path = extract_audio(video_path)
    try:
        transcript = transcribe_audio(audio_path)
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
    return transcript
