"""
Whisper Transcriber Module
Trích xuất audio từ video và tạo segments có timestamp chính xác.
"""

import os
from typing import List, Dict, Any

# Type alias cho Whisper segment
Segment = Dict[str, Any]  # {"start": float, "end": float, "text": str}


def transcribe_video(
    video_path: str,
    language: str = "vi",
    model_name: str = "small",
    fp16: bool = False,
) -> List[Segment]:
    """
    Dùng Whisper transcribe video/audio → danh sách segments có timestamp.

    Args:
        video_path: Đường dẫn tới file video/audio.
        language:   Ngôn ngữ (mặc định: vi).
        model_name: Whisper model (tiny, base, small, medium, large).
        fp16:       Dùng FP16 — nhanh hơn trên GPU, tắt cho CPU.

    Returns:
        List[Segment]: [{start, end, text}, ...]
    """
    import whisper

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video không tồn tại: {video_path}")

    print(f"  [Whisper] Loading model '{model_name}'...")
    model = whisper.load_model(model_name)

    print(f"  [Whisper] Transcribing '{os.path.basename(video_path)}'...")
    result = model.transcribe(video_path, language=language, fp16=fp16)

    segments = result.get("segments", [])
    print(f"  [Whisper] → {len(segments)} segments detected")

    return segments
