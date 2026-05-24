"""
Anam Avatar — FastAPI Web Server
Serve giao diện web, video bài giảng, phụ đề, và kết nối Anam AI Avatar.
"""

import asyncio
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import cv2
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from anam import AnamClient, AnamEvent
from anam.types import MessageStreamEvent, AgentAudioInputConfig, PersonaConfig
from dotenv import load_dotenv
import os
import uvicorn
from pathlib import Path
import httpx

# ── Resolve project paths ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WEB_DIR = PROJECT_ROOT / "web"
ASSETS_DIR = PROJECT_ROOT / "assets"
DATA_DIR = PROJECT_ROOT / "data"
SRT_DIR = DATA_DIR / "output" / "srt"
SRT_CACHE_DIR = DATA_DIR / "output" / "srt" / "cache"
SUMMARY_CACHE_DIR = DATA_DIR / "output" / "summary"
VIDEOS_DIR = DATA_DIR / "videos"
VIDEO_ITEMS_DIR = DATA_DIR / "video-item"

# ── Load environment ──────────────────────────────────────────────────────────
env_path = PROJECT_ROOT / ".env"
if not env_path.exists():
    env_path = PROJECT_ROOT / "Backend" / ".env"
load_dotenv(env_path)

ANAM_API_KEY = os.getenv("ANAM_API_KEY")
ANAM_PERSONA_ID = os.getenv("ANAM_PERSONA_ID")
HEYGEN_API_KEY = os.getenv("HEYGEN_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

if not ANAM_API_KEY or not ANAM_PERSONA_ID:
    raise ValueError("❌ Thiếu ANAM_API_KEY hoặc ANAM_PERSONA_ID trong file .env")

latest_frame: bytes | None = None
session_task = None
current_session = None
is_connected = False

audio_clients: list[WebSocket] = []
chat_clients: list[WebSocket] = []

agent_audio_stream = None

# ── Anam client — khởi tạo mặc định, sẽ được tạo lại khi start với lang ──────
# Client sẽ được tạo động trong /start theo ngôn ngữ được chọn
_anam_client: AnamClient | None = None

def make_anam_client(lang: str = "en") -> AnamClient:
    """Tạo AnamClient với language_code tương ứng."""
    persona = PersonaConfig(
        persona_id=ANAM_PERSONA_ID,
        language_code=lang,  # "en" hoặc "vi"
    )
    c = AnamClient(api_key=ANAM_API_KEY, persona_config=persona)

    @c.on(AnamEvent.MESSAGE_STREAM_EVENT_RECEIVED)
    async def on_stream(event: MessageStreamEvent):
        await broadcast_chat({
            "type": "stream",
            "id": event.id,
            "role": event.role.value,
            "content": event.content,
            "content_index": event.content_index,
            "end_of_speech": event.end_of_speech,
            "interrupted": event.interrupted,
        })

    return c


# ─── Broadcast helpers ────────────────────────────────────────────────────────

async def broadcast_audio(pcm_bytes: bytes):
    dead = []
    for ws in audio_clients:
        try:
            await ws.send_bytes(pcm_bytes)
        except Exception:
            dead.append(ws)
    for ws in dead:
        audio_clients.remove(ws)


async def broadcast_chat(data: dict):
    import json
    msg = json.dumps(data, ensure_ascii=False)
    dead = []
    for ws in chat_clients:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        chat_clients.remove(ws)


# ─── Core Anam session ────────────────────────────────────────────────────────

async def run_anam_session(lang: str = "en"):
    global latest_frame, current_session, is_connected, _anam_client
    try:
        print(f"[*] Dang ket noi Anam (lang={lang})...")
        _anam_client = make_anam_client(lang)
        async with _anam_client.connect() as session:
            current_session = session
            is_connected = True
            print("[OK] Ket noi thanh cong!")

            async def consume_video():
                global latest_frame
                async for frame in session.video_frames():
                    bgr = frame.to_ndarray(format="bgr24")
                    _, jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    latest_frame = jpeg.tobytes()

            async def consume_audio():
                async for frame in session.audio_frames():
                    pcm = frame.to_ndarray().tobytes()
                    if audio_clients:
                        await broadcast_audio(pcm)

            await asyncio.gather(consume_video(), consume_audio())

    except asyncio.CancelledError:
        print("[STOP] Session da dung.")
    except Exception as e:
        print(f"[ERR] Loi: {type(e).__name__}: {e}")
    finally:
        current_session = None
        is_connected = False
        latest_frame = None


# ─── App setup ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── HTTP endpoints ───────────────────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/logo/logo.png")
async def logo():
    logo_path = ASSETS_DIR / "logo.png"
    if not logo_path.exists():
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Logo not found")
    return FileResponse(str(logo_path), media_type="image/png")


@app.get("/subtitles/{filename}")
async def serve_subtitle(filename: str):
    if not filename.endswith(".srt"):
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Only .srt files allowed")
    sub_path = SRT_DIR / filename
    if not sub_path.exists():
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Subtitle not found")
    from fastapi.responses import Response
    from src.pipeline.cleaner import clean_text
    raw = sub_path.read_text(encoding="utf-8")
    cleaned = clean_text(raw)
    return Response(content=cleaned, media_type="text/plain; charset=utf-8")


@app.get("/lecture")
async def lecture_video():
    video_file = None
    if VIDEOS_DIR.exists():
        for f in VIDEOS_DIR.iterdir():
            if f.suffix.lower() in (".mp4", ".webm", ".mkv"):
                video_file = f
                break
    if not video_file:
        legacy_dir = PROJECT_ROOT / "video"
        if legacy_dir.exists():
            for f in legacy_dir.iterdir():
                if f.suffix.lower() in (".mp4", ".webm", ".mkv"):
                    video_file = f
                    break
    if not video_file or not video_file.exists():
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(
        str(video_file),
        media_type="video/mp4",
        headers={"Accept-Ranges": "bytes"},
    )


@app.get("/api/local-videos")
async def list_local_videos():
    items = []
    if not VIDEO_ITEMS_DIR.exists():
        return JSONResponse(content={"data": items})
    for item_dir in sorted(VIDEO_ITEMS_DIR.iterdir()):
        if not item_dir.is_dir():
            continue
        video_dir = item_dir / "video"
        if not video_dir.exists():
            continue
        video_file = None
        for f in video_dir.iterdir():
            if f.suffix.lower() in (".mp4", ".webm", ".mkv"):
                video_file = f
                break
        if not video_file:
            continue
        script_file = None
        script_dir = item_dir / "script"
        if script_dir.exists():
            for f in script_dir.iterdir():
                if f.suffix.lower() in (".txt", ".srt"):
                    script_file = f
                    break
        title = video_file.stem.replace("_", " ")
        size_mb = video_file.stat().st_size / 1024 / 1024
        items.append({
            "id": item_dir.name,
            "title": title,
            "video_url": f"http://localhost:8000/local-video/{item_dir.name}",
            "size_mb": round(size_mb, 1),
            "has_script": script_file is not None,
            "source": "local",
            "item_id": item_dir.name,
        })
    return JSONResponse(content={"data": items})


@app.get("/local-video/{item_id}")
async def serve_local_video(item_id: str):
    from fastapi import HTTPException
    item_dir = VIDEO_ITEMS_DIR / item_id
    if not item_dir.exists():
        raise HTTPException(status_code=404, detail="Video item not found")
    video_dir = item_dir / "video"
    if not video_dir.exists():
        raise HTTPException(status_code=404, detail="No video directory")
    for f in video_dir.iterdir():
        if f.suffix.lower() in (".mp4", ".webm", ".mkv"):
            return FileResponse(
                str(f),
                media_type="video/mp4",
                headers={"Accept-Ranges": "bytes"},
            )
    raise HTTPException(status_code=404, detail="No video file found")


@app.post("/start")
async def start(body: dict = {}):
    """
    Khởi động session Anam.
    Body: { "lang": "vi" | "en" }  (mặc định "en")
    """
    global session_task
    if session_task and not session_task.done():
        return {"ok": False, "message": "Đã đang chạy"}
    lang = body.get("lang", "en") if body else "en"
    if lang not in ("vi", "en"):
        lang = "en"
    print(f"[START] lang={lang}")
    session_task = asyncio.create_task(run_anam_session(lang))
    return {"ok": True, "lang": lang}


@app.post("/stop")
async def stop():
    global session_task, latest_frame, is_connected, agent_audio_stream
    if session_task:
        session_task.cancel()
        try:
            await session_task
        except asyncio.CancelledError:
            pass
        session_task = None
    latest_frame = None
    is_connected = False
    agent_audio_stream = None
    return {"ok": True}


@app.post("/send")
async def send_message(body: dict):
    if not current_session:
        return {"ok": False, "message": "Chưa kết nối"}
    text = body.get("text", "").strip()
    if not text:
        return {"ok": False, "message": "Tin nhắn trống"}
    try:
        await broadcast_chat({
            "type": "stream",
            "id": f"user::{text[:20]}",
            "role": "user",
            "content": text,
            "content_index": 0,
            "end_of_speech": True,
            "interrupted": False,
        })
        await current_session.send_message(text)
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "message": str(e)}


@app.get("/status")
async def status():
    return {"connected": is_connected}


# ─── HeyGen proxy endpoint ────────────────────────────────────────────────────

@app.get("/api/heygen/videos")
async def heygen_videos(
    limit: int = Query(default=20, ge=1, le=100),
    token: str = Query(default=None),
    title: str = Query(default=None),
    folder_id: str = Query(default=None),
):
    if not HEYGEN_API_KEY:
        return JSONResponse(
            status_code=503,
            content={"error": "HEYGEN_API_KEY chưa được cấu hình trong .env"}
        )
    params: dict = {"limit": limit}
    if token:
        params["token"] = token
    if title:
        params["title"] = title
    if folder_id:
        params["folder_id"] = folder_id
    async with httpx.AsyncClient(timeout=15.0) as client_http:
        try:
            resp = await client_http.get(
                "https://api.heygen.com/v3/videos",
                headers={"x-api-key": HEYGEN_API_KEY},
                params=params,
            )
            resp.raise_for_status()
            data = resp.json()
            if "data" in data and isinstance(data["data"], list):
                data["data"] = [
                    v for v in data["data"]
                    if v.get("status") == "completed" and v.get("video_url")
                ]
            return JSONResponse(content=data)
        except httpx.HTTPStatusError as e:
            return JSONResponse(
                status_code=e.response.status_code,
                content={"error": f"HeyGen API lỗi: {e.response.text}"}
            )
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})


# ─── Subtitle on-demand loading ───────────────────────────────────────────────

@app.post("/api/subtitles/load")
async def load_subtitles_api(body: dict):
    from concurrent.futures import ThreadPoolExecutor
    _thread_pool = ThreadPoolExecutor(max_workers=2)

    subtitle_url = body.get("subtitle_url")
    video_url = body.get("video_url")
    item_id = body.get("item_id")
    video_id = body.get("video_id")

    if item_id:
        srt_cache_dir = VIDEO_ITEMS_DIR / item_id / "srt"
        srt_cache_file = srt_cache_dir / "subtitle_vi_en.srt"
        if srt_cache_file.exists():
            from src.pipeline.cleaner import clean_text
            cached = srt_cache_file.read_text(encoding="utf-8")
            cached = clean_text(cached)
            return JSONResponse(content={"ok": True, "srt": cached, "source": "local-cache"})

        item_dir = VIDEO_ITEMS_DIR / item_id
        video_dir = item_dir / "video"
        if video_dir.exists():
            for f in video_dir.iterdir():
                if f.suffix.lower() in (".mp4", ".webm", ".mkv"):
                    import whisper
                    loop = asyncio.get_event_loop()

                    def run_local_whisper(vpath=str(f), cache_path=str(srt_cache_file), cache_dir=str(srt_cache_dir)):
                        from src.pipeline.transcriber import transcribe_video
                        from src.pipeline.srt_formatter import segments_to_bilingual_srt
                        from src.pipeline.cleaner import clean_segments
                        from src.pipeline.translator import translate_sentences
                        wm = whisper.load_model("small")
                        segments = transcribe_video(vpath, language="vi", model_name="small", whisper_model=wm)
                        segments = clean_segments(segments)
                        vi_texts = [seg["text"].strip() for seg in segments]
                        en_texts = translate_sentences(vi_texts, src="vi", dest="en", delay=0.2)
                        srt = segments_to_bilingual_srt(segments, en_texts)
                        os.makedirs(cache_dir, exist_ok=True)
                        with open(cache_path, "w", encoding="utf-8") as fp:
                            fp.write(srt)
                        return srt

                    try:
                        srt_text = await loop.run_in_executor(_thread_pool, run_local_whisper)
                        return JSONResponse(content={"ok": True, "srt": srt_text, "source": "whisper-cached"})
                    except Exception as e:
                        return JSONResponse(status_code=500, content={"ok": False, "error": f"Lỗi Whisper local: {e}"})
                    break

    if video_id and not item_id:
        heygen_cache = SRT_CACHE_DIR / f"{video_id}_vi_en.srt"
        if heygen_cache.exists():
            from src.pipeline.cleaner import clean_text
            cached = heygen_cache.read_text(encoding="utf-8")
            cached = clean_text(cached)
            return JSONResponse(content={"ok": True, "srt": cached, "source": "heygen-cache"})

    if subtitle_url:
        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as c:
                resp = await c.get(subtitle_url)
                resp.raise_for_status()
                raw_srt = resp.text
            from src.pipeline.srt_formatter import parse_srt
            cues = parse_srt(raw_srt)
            en_texts = [cue["text"].replace("\n", " ").strip() for cue in cues]
            from concurrent.futures import ThreadPoolExecutor
            pool = ThreadPoolExecutor(max_workers=1)
            loop = asyncio.get_event_loop()
            def translate_en_to_vi():
                from src.pipeline.translator import translate_sentences
                return translate_sentences(en_texts, src="en", dest="vi", delay=0.2)
            vi_texts = await loop.run_in_executor(pool, translate_en_to_vi)
            from src.pipeline.srt_formatter import format_srt_time
            srt_lines = []
            for i, (cue, vi, en) in enumerate(zip(cues, vi_texts, en_texts), 1):
                start = format_srt_time(cue["start"])
                end = format_srt_time(cue["end"])
                srt_lines.extend([f"{i}", f"{start} --> {end}", vi, en, ""])
            bilingual_srt = "\n".join(srt_lines)
            if video_id:
                os.makedirs(str(SRT_CACHE_DIR), exist_ok=True)
                (SRT_CACHE_DIR / f"{video_id}_vi_en.srt").write_text(bilingual_srt, encoding="utf-8")
            return JSONResponse(content={"ok": True, "srt": bilingual_srt, "source": "heygen"})
        except Exception as e:
            return JSONResponse(status_code=500, content={"ok": False, "error": f"Lỗi tải phụ đề HeyGen: {e}"})

    return JSONResponse(status_code=400, content={"ok": False, "error": "Cần subtitle_url, video_url hoặc item_id"})


# ─── Lecture Summarization ─────────────────────────────────────────────────────

async def _call_groq_summarize(cleaned: str, label: str, summary_cache: Path, lang: str = "vi") -> JSONResponse:
    if not GROQ_API_KEY:
        return JSONResponse(status_code=500, content={"ok": False, "error": "Chưa cấu hình GROQ_API_KEY"})
    if lang == "en":
        system_prompt = (
            "You are a professional lecture summarizer. "
            "Summarize the lecture content in English with clear structure and main topics. "
            "Use markdown format: ## headings, bullet points, and bold for key terms. "
            "Keep it concise yet comprehensive, around 300-500 words."
        )
        user_prompt = f"Please summarize the following lecture:\n\n{cleaned[:6000]}"
    else:
        system_prompt = "Bạn là trợ lý tóm tắt bài giảng. Tóm tắt bằng tiếng Việt, markdown: ## tiêu đề, bullet, bold từ khóa. ~300-500 từ."
        user_prompt = f"Hãy tóm tắt bài giảng sau:\n\n{cleaned[:6000]}"
    try:
        async with httpx.AsyncClient(timeout=60.0) as c:
            resp = await c.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                json={"model": "llama-3.3-70b-versatile", "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ], "temperature": 0.3, "max_tokens": 1500},
            )
            resp.raise_for_status()
            data = resp.json()
        summary = data["choices"][0]["message"]["content"]
        os.makedirs(str(summary_cache.parent), exist_ok=True)
        summary_cache.write_text(summary, encoding="utf-8")
        return JSONResponse(content={"ok": True, "summary": summary, "source": "groq"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": f"Lỗi tóm tắt: {e}"})


@app.post("/api/summarize")
async def summarize_lecture(body: dict):
    """
    Tóm tắt bài giảng từ script/SRT.
    Hỗ trợ cả local video (item_id) và HeyGen video (video_id).
    Sử dụng Groq API (llama-3.3-70b-versatile) hoặc cache nếu có.
    """
    item_id = body.get("item_id")
    video_id = body.get("video_id")
    lang = body.get("lang", "vi")
    if lang not in ("vi", "en"):
        lang = "vi"
    if not item_id and not video_id:
        return JSONResponse(status_code=400, content={"ok": False, "error": "Cần item_id hoặc video_id"})
    from src.pipeline.cleaner import clean_script
    if item_id:
        item_dir = VIDEO_ITEMS_DIR / item_id
        if not item_dir.exists():
            return JSONResponse(
                status_code=404,
                content={"ok": False, "error": "Video item không tồn tại"}
            )

        # Check cache (theo ngôn ngữ)
        summary_cache = item_dir / f"summary_{lang}.txt"
        if summary_cache.exists():
            print(f"[SUM] Cache hit for {item_id} ({lang})")
            return JSONResponse(content={
                "ok": True,
                "summary": summary_cache.read_text(encoding="utf-8"),
                "source": "cache",
            })

        # Read script
        script_dir = item_dir / "script"
        script_text = None
        if script_dir.exists():
            for f in script_dir.iterdir():
                if f.suffix.lower() == ".txt":
                    script_text = f.read_text(encoding="utf-8")
                    break
        if not script_text:
            srt_cache = item_dir / "srt" / "subtitle_vi_en.srt"
            if srt_cache.exists():
                script_text = srt_cache.read_text(encoding="utf-8")
        if not script_text:
            return JSONResponse(status_code=404, content={"ok": False, "error": "Không tìm thấy script. Hãy tải phụ đề trước."})
        cleaned = clean_script(script_text)
        return await _call_groq_summarize(cleaned, item_id, summary_cache, lang)
    if video_id:
        os.makedirs(str(SUMMARY_CACHE_DIR), exist_ok=True)
        summary_cache = SUMMARY_CACHE_DIR / f"{video_id}_{lang}.txt"

        # Check cache
        if summary_cache.exists():
            print(f"[SUM] HeyGen cache hit for {video_id} ({lang})")
            return JSONResponse(content={
                "ok": True,
                "summary": summary_cache.read_text(encoding="utf-8"),
                "source": "cache",
            })

        # Read from cached SRT
        srt_cache = SRT_CACHE_DIR / f"{video_id}_vi_en.srt"
        if not srt_cache.exists():
            return JSONResponse(status_code=404, content={"ok": False, "error": "Chưa có phụ đề. Hãy tải phụ đề trước."})
        cleaned = clean_script(srt_cache.read_text(encoding="utf-8"))
        return await _call_groq_summarize(cleaned, video_id, summary_cache, lang)


# ─── Audio INPUT — /mic (dùng cho cả EN và VI, Anam tự STT) ──────────────────

@app.websocket("/mic")
async def mic_ws(websocket: WebSocket):
    """
    Frontend gửi raw 16-bit PCM qua WebSocket binary frames.
    Server relay thẳng vào session.send_user_audio().
    Anam tự xử lý STT cho cả EN và VI (theo language_code đã set khi /start).
    """
    await websocket.accept()
    # /mic relay PCM thang vao Anam neu connected, neu khong thi bo qua (khong close WS)

    sample_rate  = int(websocket.query_params.get("sample_rate", 16000))
    num_channels = int(websocket.query_params.get("channels", 1))

    print(f"[MIC] Mic client ket noi: {sample_rate}Hz, {num_channels}ch")
    chunk_count = 0
    try:
        while True:
            pcm_bytes = await websocket.receive_bytes()
            chunk_count += 1
            if current_session and is_connected:
                try:
                    current_session.send_user_audio(
                        audio_bytes=pcm_bytes,
                        sample_rate=sample_rate,
                        num_channels=num_channels,
                    )
                    if chunk_count % 50 == 1:
                        print(f"[MIC] Chunk #{chunk_count}: {len(pcm_bytes)} bytes relay OK")
                except Exception as relay_err:
                    if chunk_count <= 3:
                        print(f"[MIC] send_user_audio error (chunk#{chunk_count}): {relay_err}")
            else:
                if chunk_count <= 2:
                    print(f"[MIC] Chunk #{chunk_count}: skip (session={bool(current_session)}, connected={is_connected})")
    except WebSocketDisconnect:
        print(f"[MIC] Mic client ngat ket noi (tong {chunk_count} chunks)")
    except Exception as e:
        print(f"[ERR] Loi mic_ws: {e}")


# ─── Vietnamese STT endpoint ─────────────────────────────────────────────────

@app.websocket("/stt")
async def stt_ws(websocket: WebSocket):
    """
    Nhận raw 16-bit PCM mono 16kHz từ frontend,
    gom đủ chunk rồi dùng Whisper nhận dạng tiếng Việt,
    sau đó gửi kết quả text vào Anam qua send_message().
    """
    await websocket.accept()

    if not current_session or not is_connected:
        await websocket.send_text('{"error":"Chưa kết nối Anam"}')
        await websocket.close()
        return

    sample_rate = int(websocket.query_params.get("sample_rate", 16000))
    lang = websocket.query_params.get("lang", "vi")
    chunk_ms = int(websocket.query_params.get("chunk_ms", 1500))

    bytes_needed = int(sample_rate * (chunk_ms / 1000) * 2)

    print(f"[STT-VI] STT client ket noi: {sample_rate}Hz, lang={lang}, chunk_ms={chunk_ms}")
    pcm_buffer = bytearray()

    def run_whisper(pcm_bytes: bytes) -> str:
        """Chạy Whisper trong thread pool để không block event loop."""
        audio_np = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        result = whisper_model.transcribe(audio_np, language=lang, fp16=False)
        return result["text"].strip()

    try:
        while True:
            data = await websocket.receive()

            if "bytes" in data and data["bytes"]:
                pcm_buffer.extend(data["bytes"])

                while len(pcm_buffer) >= bytes_needed:
                    chunk = bytes(pcm_buffer[:bytes_needed])
                    del pcm_buffer[:bytes_needed]

                    loop = asyncio.get_event_loop()
                    text = await loop.run_in_executor(_thread_pool, run_whisper, chunk)

                    if text and current_session:
                        print(f"[STT] Nhan dang: '{text}'")
                        await broadcast_chat({
                            "type": "stream",
                            "id": f"stt::{text[:20]}",
                            "role": "user",
                            "content": text,
                            "content_index": 0,
                            "end_of_speech": True,
                            "interrupted": False,
                        })
                        await current_session.send_message(text)

            elif "text" in data:
                cmd = (data["text"] or "").strip().lower()
                if cmd == "flush" and pcm_buffer and current_session:
                    chunk = bytes(pcm_buffer)
                    pcm_buffer.clear()
                    loop = asyncio.get_event_loop()
                    text = await loop.run_in_executor(_thread_pool, run_whisper, chunk)
                    if text:
                        print(f"[STT-FLUSH] Nhan dang: '{text}'")
                        await broadcast_chat({
                            "type": "stream",
                            "id": f"stt::{text[:20]}",
                            "role": "user",
                            "content": text,
                            "content_index": 0,
                            "end_of_speech": True,
                            "interrupted": False,
                        })
                        await current_session.send_message(text)

    except WebSocketDisconnect:
        print("[STT-VI] STT client ngat ket noi")
    except Exception as e:
        print(f"[ERR] Loi stt_ws: {e}")


@app.websocket("/agent-audio")
async def agent_audio_ws(websocket: WebSocket):
    global agent_audio_stream
    await websocket.accept()
    if not current_session or not is_connected:
        await websocket.send_text('{"error":"Chưa kết nối Anam"}')
        await websocket.close()
        return
    sample_rate = int(websocket.query_params.get("sample_rate", 24000))
    channels = int(websocket.query_params.get("channels", 1))
    encoding = websocket.query_params.get("encoding", "pcm_s16le")
    config = AgentAudioInputConfig(encoding=encoding, sample_rate=sample_rate, channels=channels)
    agent_audio_stream = current_session.create_agent_audio_input_stream(config)
    try:
        while True:
            message = await websocket.receive()
            if "bytes" in message and message["bytes"]:
                await agent_audio_stream.send_audio_chunk(message["bytes"])
            elif "text" in message:
                cmd = message["text"].strip().lower()
                if cmd == "end":
                    await agent_audio_stream.end_sequence()
                elif cmd == "close":
                    break
    except WebSocketDisconnect:
        pass
    finally:
        if agent_audio_stream:
            try:
                await agent_audio_stream.end_sequence()
            except Exception:
                pass
        agent_audio_stream = None


# ─── WebSocket output endpoints ───────────────────────────────────────────────

@app.websocket("/audio")
async def audio_ws(websocket: WebSocket):
    await websocket.accept()
    audio_clients.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in audio_clients:
            audio_clients.remove(websocket)


@app.websocket("/chat")
async def chat_ws(websocket: WebSocket):
    await websocket.accept()
    chat_clients.append(websocket)
    print(f"[CHAT] Chat client ket noi ({len(chat_clients)} clients)")
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in chat_clients:
            chat_clients.remove(websocket)


# ─── Video stream ─────────────────────────────────────────────────────────────

async def mjpeg_generator():
    boundary = b"--frame\r\n"
    content_type = b"Content-Type: image/jpeg\r\n\r\n"
    try:
        while True:
            if latest_frame:
                yield boundary + content_type + latest_frame + b"\r\n"
            await asyncio.sleep(1 / 30)
    except (asyncio.CancelledError, Exception):
        pass


@app.get("/video")
async def video_feed():
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache"},
    )


if __name__ == "__main__":
    uvicorn.run("src.server.app:app", host="0.0.0.0", port=8000, reload=False)