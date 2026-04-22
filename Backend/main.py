import asyncio
import cv2
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from anam import AnamClient, AnamEvent
from anam.types import MessageStreamEvent
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
chat_clients: list[WebSocket] = []  # WebSocket nhận lịch sử chat

client = AnamClient(api_key=ANAM_API_KEY, persona_id=ANAM_PERSONA_ID)


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
    """Gửi sự kiện chat tới tất cả chat clients."""
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


@client.on(AnamEvent.MESSAGE_STREAM_EVENT_RECEIVED)
async def on_stream(event: MessageStreamEvent):
    """Gửi từng chunk text tới frontend theo thời gian thực."""
    await broadcast_chat({
        "type": "stream",
        "id": event.id,
        "role": event.role.value,
        "content": event.content,
        "content_index": event.content_index,
        "end_of_speech": event.end_of_speech,
        "interrupted": event.interrupted,
    })


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
    global session_task, latest_frame, is_connected
    if session_task:
        session_task.cancel()
        try:
            await session_task
        except asyncio.CancelledError:
            pass
        session_task = None
    latest_frame = None
    is_connected = False
    return {"ok": True}


@app.post("/send")
async def send_message(body: dict):
    if not current_session:
        return {"ok": False, "message": "Chưa kết nối"}
    text = body.get("text", "").strip()
    if not text:
        return {"ok": False, "message": "Tin nhắn trống"}
    try:
        # Hiện tin nhắn user ngay lập tức
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
    print(f"💬 Chat client kết nối ({len(chat_clients)} clients)")
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in chat_clients:
            chat_clients.remove(websocket)


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