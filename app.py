import os
import asyncio
from fastapi import FastAPI, UploadFile
from video_processor import process_video
from summarizer import summarize_text
from comparator import choose_best

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
CHUNK_SIZE = 1024 * 1024  # 1MB chunks

async def save_file(file: UploadFile, path: str) -> None:
    """Save uploaded file to disk asynchronously in chunks."""
    with open(path, "wb") as f:
        while True:
            chunk = await file.read(CHUNK_SIZE)
            if not chunk:
                break
            f.write(chunk)

async def process_single_video(file_path: str) -> str:
    """Process a single video: transcribe & summarize."""
    transcript = await asyncio.to_thread(process_video, file_path)
    summary = await asyncio.to_thread(summarize_text, transcript)
    os.remove(file_path)
    return summary

@app.post("/compare_videos/")
async def compare_videos(topic: str, files: list[UploadFile]):
    video_names = []
    file_paths = []
    save_tasks = []
    for file in files:
        if not file.filename:
            raise ValueError("Uploaded file has no filename")
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        video_names.append(file.filename)
        file_paths.append(file_path)
        save_tasks.append(save_file(file, file_path))
    await asyncio.gather(*save_tasks)
    # Process all videos in parallel
    process_tasks = [process_single_video(path) for path in file_paths]
    summaries = await asyncio.gather(*process_tasks)
    # Compare summaries and choose best video
    best_index, scores = choose_best(summaries, topic)
    return {
        "topic": topic,
        "videos": [
            {"name": name, "summary": summ, "score": score}
            for name, summ, score in zip(video_names, summaries, scores)
        ],
        "best_video": video_names[best_index],
    }
