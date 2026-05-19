"""
Script Cleaner Module
Làm sạch SSML tags, emotion tags, chuẩn hóa từ vựng cho script bài giảng.

Functions:
    - clean_text:        Chuẩn hóa phonetic, SSML tags cho text thô
    - clean_script:      Clean toàn bộ script text
    - clean_segments:    Clean danh sách segments (Whisper output)
    - split_into_sentences: Tách script thành danh sách câu
"""

import re
from typing import List, Dict, Any


# ── Phonetic normalization rules ──────────────────────────────────────────────
# Bản đồ chuẩn hóa các lỗi phiên âm phổ biến từ Whisper/TTS
PHONETIC_MAP = {
    # Viettel variants
    r"\bViet\s*Theo\b": "Viettel",
    r"\bViệt\s*Theo\b": "Viettel",
    r"\bViệt\s*Tel\b": "Viettel",
    r"\bViet\s*Tel\b": "Viettel",
    r"\bViệt\s*Teo\b": "Viettel",
    r"\bViet\s*Teo\b": "Viettel",
    r"\bViệt\s*Thế\s*Ồ\b": "Viettel",
    r"\bViettel\s*Telecom\b": "Viettel Telecom",
    # Common TTS/Whisper errors
    r"\bAI\s*Ây\b": "AI",
    r"\bA\.I\.\b": "AI",
}

# SSML tags to strip
SSML_PATTERN = re.compile(r"<[^>]+>")

# Emotion tags: [happy], [sad], [excited], etc.
EMOTION_PATTERN = re.compile(r"\[(?:happy|sad|excited|angry|neutral|pause|breath)\]", re.IGNORECASE)

# Repeated whitespace
MULTI_SPACE = re.compile(r"[ \t]+")
MULTI_NEWLINE = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """
    Chuẩn hóa text: xóa SSML tags, emotion markers, sửa lỗi phiên âm.

    Args:
        text: Text thô cần clean.

    Returns:
        Text đã được chuẩn hóa.
    """
    if not text:
        return ""

    # 1. Strip SSML tags
    text = SSML_PATTERN.sub("", text)

    # 2. Strip emotion tags
    text = EMOTION_PATTERN.sub("", text)

    # 3. Phonetic normalization
    for pattern, replacement in PHONETIC_MAP.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # 4. Clean whitespace
    text = MULTI_SPACE.sub(" ", text)
    text = MULTI_NEWLINE.sub("\n\n", text)

    # 5. Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    return text.strip()


def clean_script(raw_text: str) -> str:
    """
    Clean toàn bộ script text:
    - Xóa SSML tags
    - Chuẩn hóa phiên âm
    - Giữ cấu trúc đoạn văn

    Args:
        raw_text: Script text thô.

    Returns:
        Script đã clean.
    """
    if not raw_text:
        return ""

    # Basic cleaning
    cleaned = clean_text(raw_text)

    # Remove empty lines between paragraphs (keep structure)
    paragraphs = [p.strip() for p in cleaned.split("\n\n") if p.strip()]
    cleaned = "\n\n".join(paragraphs)

    return cleaned


def clean_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Clean danh sách segments từ Whisper output.
    Mỗi segment có format: {start, end, text, ...}

    Args:
        segments: Danh sách segments cần clean.

    Returns:
        Danh sách segments đã clean.
    """
    cleaned = []
    for seg in segments:
        new_seg = dict(seg)  # Copy to avoid mutation
        new_seg["text"] = clean_text(seg.get("text", ""))
        if new_seg["text"]:  # Skip empty segments
            cleaned.append(new_seg)
    return cleaned


def split_into_sentences(text: str) -> List[str]:
    """
    Tách script text thành danh sách câu.
    Sử dụng dấu chấm câu tiếng Việt: . ? ! và newline.

    Args:
        text: Script text đã clean.

    Returns:
        Danh sách các câu riêng lẻ.
    """
    if not text:
        return []

    # Split on sentence-ending punctuation
    # Keep the punctuation with the sentence
    raw_sentences = re.split(r'(?<=[.!?])\s+', text)

    sentences = []
    for s in raw_sentences:
        s = s.strip()
        if not s:
            continue
        # Further split on newlines if sentence is too long
        for line in s.split("\n"):
            line = line.strip()
            if line:
                sentences.append(line)

    return sentences
