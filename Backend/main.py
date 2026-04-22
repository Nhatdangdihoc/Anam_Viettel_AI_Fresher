import asyncio
import cv2
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from anam import AnamClient, AnamEvent
from anam.types import MessageStreamEvent, AgentAudioInputConfig
from dotenv import load_dotenv
import os
import uvicorn

load_dotenv()
ANAM_API_KEY    = os.getenv("ANAM_API_KEY")
ANAM_PERSONA_ID = os.getenv("ANAM_PERSONA_ID")

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
            print("✅ Kết nối thành công!")

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

    print(f"🎤 Mic client kết nối: {sample_rate}Hz, {num_channels}ch")
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
        print("🎤 Mic client ngắt kết nối")
    except Exception as e:
        print(f"❌ Lỗi mic_ws: {e}")


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
    print(f"🔊 AgentAudio stream mở: {sample_rate}Hz, {channels}ch, {encoding}")

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
                    print("🔊 AgentAudio: end_sequence()")
                elif cmd == "close":
                    break

    except WebSocketDisconnect:
        print("🔊 AgentAudio client ngắt kết nối")
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
    print(f"💬 Chat client kết nối ({len(chat_clients)} clients)")
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