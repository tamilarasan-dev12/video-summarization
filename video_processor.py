import os
import whisper
from moviepy.editor import VideoFileClip

MODEL = whisper.load_model("base")

def extract_audio(video_path: str, audio_path: str = "temp_audio.wav") -> str:
    """Extract audio from video and save as WAV."""
    clip = VideoFileClip(video_path)
    if clip.audio is None:
        clip.close()
        raise ValueError(f"No audio track found in {video_path}")
    clip.audio.write_audiofile(audio_path, verbose=False, logger=None)
    clip.close()
    return audio_path

def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio file using Whisper and return transcript as string."""
    result = MODEL.transcribe(audio_path)
    text = result["text"]
    if isinstance(text, list):
        text = " ".join(map(str, text))
    return str(text)

def process_video(video_path: str) -> str:
    """Extract audio, transcribe it, and return transcript as string."""
    audio_path = extract_audio(video_path)
    transcript = transcribe_audio(audio_path)
    os.remove(audio_path)
    return transcript
