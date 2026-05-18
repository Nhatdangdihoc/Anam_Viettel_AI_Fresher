"""
Anam Avatar — FastAPI Web Server
Serve giao diện web, video bài giảng, phụ đề, và kết nối Anam AI Avatar.
"""

import asyncio
import sys

# Fix Windows console encoding (cp1252 can't handle Unicode like →, 🔗, etc.)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import cv2
import numpy as np
import whisper
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from anam import AnamClient, AnamEvent
from anam.types import MessageStreamEvent, AgentAudioInputConfig
from dotenv import load_dotenv
import os
import uvicorn
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import httpx

# ── Resolve project paths ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WEB_DIR = PROJECT_ROOT / "web"
ASSETS_DIR = PROJECT_ROOT / "assets"
DATA_DIR = PROJECT_ROOT / "data"
SRT_DIR = DATA_DIR / "output" / "srt"
VIDEOS_DIR = DATA_DIR / "videos"
VIDEO_ITEMS_DIR = DATA_DIR / "video-item"

# ── Load environment ──────────────────────────────────────────────────────────
# Try root .env first, fallback to legacy Backend/.env
env_path = PROJECT_ROOT / ".env"
if not env_path.exists():
    env_path = PROJECT_ROOT / "Backend" / ".env"
load_dotenv(env_path)

ANAM_API_KEY = os.getenv("ANAM_API_KEY")
ANAM_PERSONA_ID = os.getenv("ANAM_PERSONA_ID")
HEYGEN_API_KEY = os.getenv("HEYGEN_API_KEY", "")

if not ANAM_API_KEY or not ANAM_PERSONA_ID:
    raise ValueError("❌ Thiếu ANAM_API_KEY hoặc ANAM_PERSONA_ID trong file .env")

latest_frame: bytes | None = None
session_task = None
current_session = None
is_connected = False

audio_clients: list[WebSocket] = []
chat_clients: list[WebSocket] = []

# AgentAudioInputStream hiện tại (dùng cho cách 2)
agent_audio_stream = None

# Whisper STT setup
print("[*] Dang tai Whisper model (small)...")
whisper_model = whisper.load_model("small")
print("[OK] Whisper model san sang.")
_thread_pool = ThreadPoolExecutor(max_workers=2)

client = AnamClient(api_key=ANAM_API_KEY, persona_id=ANAM_PERSONA_ID)


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


# ─── Anam event handler ───────────────────────────────────────────────────────

@client.on(AnamEvent.MESSAGE_STREAM_EVENT_RECEIVED)
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


# ─── Core Anam session ────────────────────────────────────────────────────────

async def run_anam_session():
    global latest_frame, current_session, is_connected
    try:
        print("[*] Dang ket noi Anam...")
        async with client.connect() as session:
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
    """Serve subtitle .srt files from the data/output/srt directory (cleaned)."""
    if not filename.endswith(".srt"):
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Only .srt files allowed")
    sub_path = SRT_DIR / filename
    if not sub_path.exists():
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Subtitle not found")
    # Apply phonetic cleaning (e.g. "Viet Theo" -> "Viettel")
    from fastapi.responses import Response
    from src.pipeline.cleaner import clean_text
    raw = sub_path.read_text(encoding="utf-8")
    cleaned = clean_text(raw)
    return Response(content=cleaned, media_type="text/plain; charset=utf-8")


@app.get("/lecture")
async def lecture_video():
    """Serve file video bài giảng."""
    # Tìm file video đầu tiên trong thư mục videos
    video_file = None
    if VIDEOS_DIR.exists():
        for f in VIDEOS_DIR.iterdir():
            if f.suffix.lower() in (".mp4", ".webm", ".mkv"):
                video_file = f
                break

    # Fallback: tìm trong thư mục video cũ
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
    """List tat ca video-item tu data/video-item/."""
    items = []
    if not VIDEO_ITEMS_DIR.exists():
        return JSONResponse(content={"data": items})

    for item_dir in sorted(VIDEO_ITEMS_DIR.iterdir()):
        if not item_dir.is_dir():
            continue
        video_dir = item_dir / "video"
        if not video_dir.exists():
            continue
        # Tim file video
        video_file = None
        for f in video_dir.iterdir():
            if f.suffix.lower() in (".mp4", ".webm", ".mkv"):
                video_file = f
                break
        if not video_file:
            continue

        # Tim script
        script_file = None
        script_dir = item_dir / "script"
        if script_dir.exists():
            for f in script_dir.iterdir():
                if f.suffix.lower() in (".txt", ".srt"):
                    script_file = f
                    break

        # Tao title tu ten file video
        title = video_file.stem.replace("_", " ")
        size_mb = video_file.stat().st_size / 1024 / 1024

        items.append({
            "id": item_dir.name,
            "title": title,
            "video_url": f"http://localhost:8000/local-video/{item_dir.name}",
            "size_mb": round(size_mb, 1),
            "has_script": script_file is not None,
            "source": "local",
        })

    return JSONResponse(content={"data": items})


@app.get("/local-video/{item_id}")
async def serve_local_video(item_id: str):
    """Serve video file tu data/video-item/{item_id}/video/."""
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
async def start():
    global session_task
    if session_task and not session_task.done():
        return {"ok": False, "message": "Đã đang chạy"}
    session_task = asyncio.create_task(run_anam_session())
    return {"ok": True}


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
    """
    Proxy lấy danh sách video từ HeyGen API.
    Chỉ trả về các video có status='completed'.
    """
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
            # Lọc chỉ giữ video đã hoàn thành
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
            return JSONResponse(
                status_code=500,
                content={"error": str(e)}
            )


# ─── Subtitle on-demand loading ───────────────────────────────────────────────

@app.post("/api/subtitles/load")
async def load_subtitles_api(body: dict):
    """
    Load phụ đề on-demand cho video đang phát.
    - Neu co subtitle_url (HeyGen translated) -> fetch + dich EN->VI
    - Neu chi co video_url -> download + Whisper transcribe (cham)
    """
    subtitle_url = body.get("subtitle_url")
    video_url = body.get("video_url")

    # Cach 1: Fetch SRT tu HeyGen -> parse + dich EN->VI -> tra cues bilingual
    if subtitle_url:
        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as c:
                resp = await c.get(subtitle_url)
                resp.raise_for_status()
                raw_srt = resp.text

            # Parse SRT (HeyGen trả về monolingual EN)
            from src.pipeline.srt_formatter import parse_srt

            cues = parse_srt(raw_srt)
            en_texts = [cue["text"].replace("\n", " ").strip() for cue in cues]

            # Dich EN -> VI trong thread pool
            loop = asyncio.get_event_loop()

            def translate_en_to_vi():
                from src.pipeline.translator import translate_sentences
                return translate_sentences(en_texts, src="en", dest="vi", delay=0.2)

            print(f"[SUB] Translating {len(en_texts)} HeyGen cues EN->VI...")
            vi_texts = await loop.run_in_executor(_thread_pool, translate_en_to_vi)

            # Tạo bilingual SRT
            from src.pipeline.srt_formatter import format_srt_time
            srt_lines = []
            for i, (cue, vi, en) in enumerate(zip(cues, vi_texts, en_texts), 1):
                start = format_srt_time(cue["start"])
                end = format_srt_time(cue["end"])
                srt_lines.extend([f"{i}", f"{start} --> {end}", vi, en, ""])

            bilingual_srt = "\n".join(srt_lines)
            print(f"[SUB] Done! {len(cues)} bilingual cues ready")

            return JSONResponse(content={
                "ok": True,
                "srt": bilingual_srt,
                "source": "heygen",
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            return JSONResponse(
                status_code=500,
                content={"ok": False, "error": f"Lỗi tải phụ đề HeyGen: {e}"}
            )

    # Cách 2: Download video + Whisper transcribe (chậm, vài phút)
    if video_url:
        import tempfile

        try:
            print(f"[SUB] Downloading video for Whisper: {video_url[:80]}...")
            async with httpx.AsyncClient(timeout=300.0, follow_redirects=True) as c:
                resp = await c.get(video_url)
                resp.raise_for_status()

            # Lưu tạm
            tmp = tempfile.NamedTemporaryFile(
                suffix=".mp4", delete=False, dir=str(DATA_DIR)
            )
            tmp.write(resp.content)
            tmp.close()
            tmp_path = tmp.name

            print(f"[SUB] Downloaded {len(resp.content) / 1024 / 1024:.1f}MB -> Whisper...")

            # Whisper trong thread pool (không block event loop)
            loop = asyncio.get_event_loop()

            def run_whisper_transcribe():
                from src.pipeline.transcriber import transcribe_video
                from src.pipeline.srt_formatter import segments_to_bilingual_srt
                from src.pipeline.cleaner import clean_segments
                from src.pipeline.translator import translate_sentences

                segments = transcribe_video(
                    tmp_path, language="vi", model_name="small"
                )
                segments = clean_segments(segments)
                
                # Translate VI -> EN
                vi_texts = [seg["text"].strip() for seg in segments]
                print(f"[SUB] Translating {len(vi_texts)} Whisper segments VI->EN...")
                en_texts = translate_sentences(vi_texts, src="vi", dest="en", delay=0.2)
                
                return segments_to_bilingual_srt(segments, en_texts)

            srt_text = await loop.run_in_executor(
                _thread_pool, run_whisper_transcribe
            )

            # Cleanup temp file
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

            print(f"[SUB] Whisper done! Generated SRT")
            return JSONResponse(content={
                "ok": True,
                "srt": srt_text,
                "source": "whisper",
            })

        except Exception as e:
            print(f"[SUB] Whisper error: {e}")
            return JSONResponse(
                status_code=500,
                content={"ok": False, "error": f"Lỗi Whisper: {e}"}
            )

    return JSONResponse(
        status_code=400,
        content={"ok": False, "error": "Cần subtitle_url hoặc video_url"}
    )


# ─── Audio INPUT endpoints ────────────────────────────────────────────────────

@app.websocket("/mic")
async def mic_ws(websocket: WebSocket):
    """
    Cách 1 – WebRTC UserAudioInputTrack.
    Frontend gửi raw 16-bit PCM qua WebSocket binary frames.
    Server relay thẳng vào session.send_user_audio().
    """
    await websocket.accept()

    if not current_session or not is_connected:
        await websocket.send_text('{"error":"Chưa kết nối Anam"}')
        await websocket.close()
        return

    sample_rate = int(websocket.query_params.get("sample_rate", 16000))
    num_channels = int(websocket.query_params.get("channels", 1))

    print(f"[MIC] Mic client ket noi: {sample_rate}Hz, {num_channels}ch")
    try:
        while True:
            pcm_bytes = await websocket.receive_bytes()
            current_session.send_user_audio(
                audio_bytes=pcm_bytes,
                sample_rate=sample_rate,
                num_channels=num_channels,
            )
    except WebSocketDisconnect:
        print("[MIC] Mic client ngat ket noi")
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
    """
    Cách 2 – AgentAudioInputStream qua WebSocket signalling.
    Phù hợp khi bạn muốn gửi audio đã xử lý vào agent.
    """
    global agent_audio_stream
    await websocket.accept()

    if not current_session or not is_connected:
        await websocket.send_text('{"error":"Chưa kết nối Anam"}')
        await websocket.close()
        return

    sample_rate = int(websocket.query_params.get("sample_rate", 24000))
    channels = int(websocket.query_params.get("channels", 1))
    encoding = websocket.query_params.get("encoding", "pcm_s16le")

    config = AgentAudioInputConfig(
        encoding=encoding,
        sample_rate=sample_rate,
        channels=channels,
    )

    agent_audio_stream = current_session.create_agent_audio_input_stream(config)
    print(f"[AUDIO] AgentAudio stream mo: {sample_rate}Hz, {channels}ch, {encoding}")

    try:
        while True:
            message = await websocket.receive()

            if "bytes" in message and message["bytes"]:
                await agent_audio_stream.send_audio_chunk(message["bytes"])

            elif "text" in message:
                cmd = message["text"].strip().lower()
                if cmd == "end":
                    await agent_audio_stream.end_sequence()
                    print("[AUDIO] AgentAudio: end_sequence()")
                elif cmd == "close":
                    break

    except WebSocketDisconnect:
        print("[AUDIO] AgentAudio client ngat ket noi")
    except Exception as e:
        print(f"[ERR] Loi agent_audio_ws: {e}")
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
    """Output: nhận PCM audio từ avatar."""
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
    """Output: nhận stream text hội thoại."""
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
