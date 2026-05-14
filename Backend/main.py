import asyncio
import cv2
import numpy as np
import whisper
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from anam import AnamClient, AnamEvent
from anam.types import MessageStreamEvent, AgentAudioInputConfig
from dotenv import load_dotenv
import os
import uvicorn
from concurrent.futures import ThreadPoolExecutor
import httpx

load_dotenv()
ANAM_API_KEY    = os.getenv("ANAM_API_KEY")
ANAM_PERSONA_ID = os.getenv("ANAM_PERSONA_ID")
HEYGEN_API_KEY  = os.getenv("HEYGEN_API_KEY", "")

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

# ─── Whisper STT setup ────────────────────────────────────────────────────────
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
        print("🔗 Đang kết nối Anam...")
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
        print("🛑 Session đã dừng.")
    except Exception as e:
        print(f"❌ Lỗi: {type(e).__name__}: {e}")
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
    frontend_path = os.path.join(os.path.dirname(__file__), "..", "Frontend", "index.html")
    return FileResponse(frontend_path)


@app.get("/logo/logo.png")
async def logo():
    logo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logo", "logo.png"))
    if not os.path.exists(logo_path):
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Logo not found")
    return FileResponse(logo_path, media_type="image/png")


@app.get("/subtitles/{filename}")
async def serve_subtitle(filename: str):
    """Serve subtitle .srt files from the video directory."""
    if not filename.endswith(".srt"):
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Only .srt files allowed")
    sub_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "video", filename))
    if not os.path.exists(sub_path):
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Subtitle not found")
    return FileResponse(sub_path, media_type="text/plain; charset=utf-8")


@app.get("/lecture")
async def lecture_video():
    """Serve file video bai giang truoc khi co live avatar."""
    video_path = os.path.join(os.path.dirname(__file__), "..", "video",
                              "L\u1ecbch s\u1eed ph\u00e1t tri\u1ec3n Viettel - Giai \u0111o\u1ea1n 3 (2010-2020)_1080p.mp4")
    video_path = os.path.abspath(video_path)
    if not os.path.exists(video_path):
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(
        video_path,
        media_type="video/mp4",
        headers={"Accept-Ranges": "bytes"},
    )


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
    Endpoint: GET https://api.heygen.com/v3/videos
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


# ─── Audio INPUT endpoints ────────────────────────────────────────────────────

@app.websocket("/mic")
async def mic_ws(websocket: WebSocket):
    """
    Cách 1 – WebRTC UserAudioInputTrack.
    Frontend gửi raw 16-bit PCM qua WebSocket binary frames.
    Server relay thẳng vào session.send_user_audio().

    Tham số query string:
        ?sample_rate=16000   (mặc định 16000)
        ?channels=1          (mặc định 1 = mono)

    Ví dụ JS phía client:
        const ws = new WebSocket("ws://localhost:8000/mic?sample_rate=16000&channels=1");
        // Dùng AudioWorkletProcessor để ghi mic rồi ws.send(pcmBuffer)
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
            # Relay trực tiếp vào Anam qua WebRTC UserAudioInputTrack
            current_session.send_user_audio(
                audio_bytes=pcm_bytes,
                sample_rate=sample_rate,
                num_channels=num_channels,
            )
    except WebSocketDisconnect:
        print("[MIC] Mic client ngat ket noi")
    except Exception as e:
        print(f"❌ Lỗi mic_ws: {e}")


# ─── Vietnamese STT endpoint ─────────────────────────────────────────────────

@app.websocket("/stt")
async def stt_ws(websocket: WebSocket):
    """
    Nhận raw 16-bit PCM mono 16kHz từ frontend,
    gom đủ chunk rồi dùng Whisper nhận dạng tiếng Việt,
    sau đó gửi kết quả text vào Anam qua send_message().

    Query params:
        ?sample_rate=16000   (mặc định 16000)
        ?lang=vi             (mặc định vi)
        ?chunk_ms=1500       (tích luỹ bao nhiêu ms rồi mới transcribe)
    """
    await websocket.accept()

    if not current_session or not is_connected:
        await websocket.send_text('{"error":"Chưa kết nối Anam"}')
        await websocket.close()
        return

    sample_rate = int(websocket.query_params.get("sample_rate", 16000))
    lang        = websocket.query_params.get("lang", "vi")
    chunk_ms    = int(websocket.query_params.get("chunk_ms", 1500))

    # Số bytes cần gom trước khi chạy STT (16-bit = 2 bytes/sample)
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

            # Nhận binary PCM chunks
            if "bytes" in data and data["bytes"]:
                pcm_buffer.extend(data["bytes"])

                # Khi đủ dữ liệu thì chạy STT
                while len(pcm_buffer) >= bytes_needed:
                    chunk = bytes(pcm_buffer[:bytes_needed])
                    del pcm_buffer[:bytes_needed]

                    loop = asyncio.get_event_loop()
                    text = await loop.run_in_executor(_thread_pool, run_whisper, chunk)

                    if text and current_session:
                        print(f"[STT] Nhan dang: '{text}'")
                        # Hiển thị text người dùng trong chat
                        await broadcast_chat({
                            "type": "stream",
                            "id": f"stt::{text[:20]}",
                            "role": "user",
                            "content": text,
                            "content_index": 0,
                            "end_of_speech": True,
                            "interrupted": False,
                        })
                        # Gửi vào Anam
                        await current_session.send_message(text)

            # Nhận text command
            elif "text" in data:
                cmd = (data["text"] or "").strip().lower()
                if cmd == "flush" and pcm_buffer and current_session:
                    # Flush phần còn lại
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
        print(f"❌ Lỗi stt_ws: {e}")


@app.websocket("/agent-audio")
async def agent_audio_ws(websocket: WebSocket):
    """
    Cách 2 – AgentAudioInputStream qua WebSocket signalling.
    Phù hợp khi bạn muốn gửi audio đã xử lý (ví dụ: TTS từ bên ngoài,
    hoặc audio file) vào agent thay vì microphone thời gian thực.

    Protocol:
        - Binary frame  → PCM chunk (base64-encode nội bộ, gửi lên Anam)
        - Text "end"    → kết thúc sequence (reset turn)
        - Text "close"  → đóng stream

    Query params:
        ?sample_rate=24000   (mặc định 24000)
        ?channels=1          (mặc định 1)
        ?encoding=pcm_s16le  (mặc định pcm_s16le)
    """
    global agent_audio_stream
    await websocket.accept()

    if not current_session or not is_connected:
        await websocket.send_text('{"error":"Chưa kết nối Anam"}')
        await websocket.close()
        return

    sample_rate = int(websocket.query_params.get("sample_rate", 24000))
    channels    = int(websocket.query_params.get("channels", 1))
    encoding    = websocket.query_params.get("encoding", "pcm_s16le")

    config = AgentAudioInputConfig(
        encoding=encoding,
        sample_rate=sample_rate,
        channels=channels,
    )

    # Tạo stream mới (hoặc dùng lại nếu đã có)
    agent_audio_stream = current_session.create_agent_audio_input_stream(config)
    print(f"[AUDIO] AgentAudio stream mo: {sample_rate}Hz, {channels}ch, {encoding}")

    try:
        while True:
            message = await websocket.receive()

            if "bytes" in message and message["bytes"]:
                # Nhận PCM chunk → gửi lên Anam
                await agent_audio_stream.send_audio_chunk(message["bytes"])

            elif "text" in message:
                cmd = message["text"].strip().lower()
                if cmd == "end":
                    # Kết thúc lượt nói hiện tại
                    await agent_audio_stream.end_sequence()
                    print("[AUDIO] AgentAudio: end_sequence()")
                elif cmd == "close":
                    break

    except WebSocketDisconnect:
        print("[AUDIO] AgentAudio client ngat ket noi")
    except Exception as e:
        print(f"❌ Lỗi agent_audio_ws: {e}")
    finally:
        # Đảm bảo kết thúc sequence sạch khi disconnect
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
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)