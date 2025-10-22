import os
import asyncio
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Tuple
import tempfile
import shutil
import uuid
import pathlib
import contextlib
import logging
from video_processor import process_video
from summarizer import summarize_text
from comparator import choose_best

logger = logging.getLogger(__name__)
app = FastAPI()

# Serve static UI using absolute paths
BASE_DIR = pathlib.Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
print("Serving UI from:", STATIC_DIR)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/debug-index", response_class=HTMLResponse)
def debug_index():
    p = STATIC_DIR / "index.html"
    try:
        html = p.read_text(encoding="utf-8")
        print("index.html bytes:", len(html.encode("utf-8")))
        return HTMLResponse(content=html, status_code=200)
    except Exception as e:
        return HTMLResponse(content=f"Error reading {p}: {e}", status_code=500)

@app.get("/", include_in_schema=False)
def root():
    return FileResponse(str(STATIC_DIR / "index.html"))

# Configuration
# Use system temp to avoid OneDrive/AV locking issues on Windows
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "video_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
CHUNK_SIZE = 1024 * 1024  # 1MB chunks

# Helpers
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

# Models
class URLCompareRequest(BaseModel):
    topic: str
    urls: List[str]

# Download helpers
def _download_with_yt_dlp(url: str, out_dir: str) -> Tuple[str, str]:
    """Download video using yt_dlp into out_dir. Returns (file_path, display_name)."""
    from yt_dlp import YoutubeDL
    import time
    tmp_id = uuid.uuid4().hex
    outtmpl = os.path.join(out_dir, f"{tmp_id}.%(ext)s")
    ydl_opts = {
        "outtmpl": outtmpl,
        # Prefer video+audio; fallback to best with container mp4 if possible
        # This avoids grabbing video-only streams (common on Shorts)
        "format": "bv*+ba/b",
        "merge_output_format": "mp4",
        "prefer_ffmpeg": True,
        "noplaylist": True,
        "quiet": True,
        "nocheckcertificate": True,
        # Networking hardening
        "forceipv4": True,
        "retries": 15,
        "fragment_retries": 15,
        "socket_timeout": 30,
        "concurrent_fragments": 1,
        "http_chunk_size": 10 * 1024 * 1024,  # 10MB
        "user_agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
        ),
        # Try multiple player clients to avoid SABR-only formats
        "extractor_args": {"youtube": {"player_client": ["android", "web"]}},
        # Use temp dir for fragments and avoid network-side temp leftovers
        "paths": {"home": out_dir, "temp": out_dir},
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        ext = info.get("ext") or "mp4"
        expected = pathlib.Path(out_dir) / f"{tmp_id}.{ext}"
        display = info.get("title") or expected.name
        # If the expected file is not present due to a Windows rename lock, try to find and rename.
        if not expected.exists():
            # Retry a few times to let external processes release the handle (OneDrive/AV)
            last_found = None
            for _ in range(10):
                matches = list(pathlib.Path(out_dir).glob(f"{tmp_id}.*"))
                last_found = matches[0] if matches else None
                if last_found and last_found.exists():
                    try:
                        last_found.replace(expected)
                        break
                    except Exception:
                        time.sleep(0.3)
                        continue
                time.sleep(0.2)
        file_path = str(expected if expected.exists() else (last_found if last_found else expected))
        return file_path, display

async def download_video_async(url: str, out_dir: str) -> Tuple[str, str]:
    return await asyncio.to_thread(_download_with_yt_dlp, url, out_dir)

# Endpoints
@app.post("/compare_videos/")
async def compare_videos(topic: str = Form(...), files: list[UploadFile] = Form(...)):
    video_names: list[str] = []
    file_paths: list[str] = []
    save_tasks = []
    for file in files:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Uploaded file has no filename")
        ext = pathlib.Path(file.filename).suffix or ".mp4"
        tmp_name = uuid.uuid4().hex + ext
        file_path = os.path.join(tempfile.gettempdir(), tmp_name)
        video_names.append(file.filename)
        file_paths.append(file_path)
        save_tasks.append(save_file(file, file_path))
    await asyncio.gather(*save_tasks)

    async def safe_process(path: str):
        try:
            return await process_single_video(path)
        except Exception as e:
            logger.exception("Processing failed for %s", path)
            with contextlib.suppress(FileNotFoundError):
                os.remove(path)
            return f"ERROR: {e}"

    summaries = await asyncio.gather(*[safe_process(p) for p in file_paths])
    valid_indices = [i for i, s in enumerate(summaries) if not str(s).startswith("ERROR:")]
    if not valid_indices:
        raise HTTPException(status_code=500, detail="All videos failed to process.")
    filtered_summaries = [summaries[i] for i in valid_indices]
    filtered_names = [video_names[i] for i in valid_indices]
    best_rel, scores, details = choose_best(filtered_summaries, topic)
    return {
        "topic": topic,
        "videos": [
            {"name": name, "summary": summ, "score": score, "details": det}
            for name, summ, score, det in zip(filtered_names, filtered_summaries, scores, details)
        ],
        "skipped": [
            {"name": video_names[i], "error": summaries[i]}
            for i in range(len(summaries)) if i not in valid_indices
        ],
        "best_video": filtered_names[best_rel],
    }

@app.post("/compare_videos_urls")
async def compare_videos_urls(payload: URLCompareRequest):
    if not payload.urls:
        raise HTTPException(status_code=400, detail="No URLs provided")
    # Create request-specific temp directory in system temp (avoid OneDrive locks)
    req_dir = tempfile.mkdtemp(prefix="dl_")
    try:
        downloads = await asyncio.gather(*[download_video_async(u, req_dir) for u in payload.urls], return_exceptions=True)
        names: list[str] = []
        paths: list[str] = []
        errors: list[dict] = []
        for i, d in enumerate(downloads):
            if isinstance(d, Exception):
                errors.append({"url": payload.urls[i], "error": str(d)})
                continue
            path, title = d
            names.append(title)
            paths.append(path)
        if not paths:
            raise HTTPException(status_code=502, detail={"message": "Failed to download all URLs", "errors": errors})

        # Make unlocked copies to avoid file locks on Windows/OneDrive
        unlocked_paths: list[str] = []
        for p in paths:
            try:
                dst = os.path.join(tempfile.gettempdir(), f"proc_{uuid.uuid4().hex}{pathlib.Path(p).suffix or '.mp4'}")
                shutil.copyfile(p, dst)
                unlocked_paths.append(dst)
            except Exception as e:
                errors.append({"name": pathlib.Path(p).name, "error": f"copy_failed: {e}"})
        paths = unlocked_paths if unlocked_paths else paths

        async def safe_process(path: str):
            try:
                return await process_single_video(path)
            except Exception as e:
                logger.exception("Processing failed for %s", path)
                with contextlib.suppress(FileNotFoundError):
                    os.remove(path)
                return f"ERROR: {e}"

        summaries = await asyncio.gather(*[safe_process(p) for p in paths])
        valid_indices = [i for i, s in enumerate(summaries) if not str(s).startswith("ERROR:")]
        if not valid_indices:
            raise HTTPException(status_code=500, detail={"message": "All processed videos failed", "download_errors": errors})
        filtered_summaries = [summaries[i] for i in valid_indices]
        filtered_names = [names[i] for i in valid_indices]
        best_rel, scores, details = choose_best(filtered_summaries, payload.topic)
        return {
            "topic": payload.topic,
            "videos": [
                {"name": n, "summary": s, "score": sc, "details": det}
                for n, s, sc, det in zip(filtered_names, filtered_summaries, scores, details)
            ],
            "skipped": errors + [
                {"name": names[i], "error": summaries[i]}
                for i in range(len(summaries)) if i not in valid_indices
            ],
            "best_video": filtered_names[best_rel],
        }
    finally:
        with contextlib.suppress(Exception):
            shutil.rmtree(req_dir, ignore_errors=True)

