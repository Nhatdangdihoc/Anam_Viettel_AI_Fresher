"""
Script Cleaner Module
Làm sạch SSML tags, emotion tags, và chuẩn hóa từ vựng từ HeyGen scripts.
"""

import re
from typing import Dict, List, Optional


# ============================================================================
# TỪ ĐIỂN CHUẨN HÓA PHÁT ÂM → CHỮ VIẾT TẮT
# Bao gồm các biến thể từ cả HeyGen script và Whisper transcription.
# ============================================================================
PHONETIC_MAP: Dict[str, str] = {
    # Viettel — biến thể phát âm phổ biến nhất từ Whisper/HeyGen
    r"\bViệt theo\b": "Viettel",
    r"\bViệt Theo\b": "Viettel",
    r"\bviệt theo\b": "Viettel",
    r"\bViệt thêu\b": "Viettel",
    r"\bViệt tel\b": "Viettel",
    r"\bViệt ten\b": "Viettel",
    r"\bVi[eê]t\s+[Tt]h?eo\b": "Viettel",
    # Viễn thông
    r"\b4 gờ\b": "4G",
    r"\b3 gờ\b": "3G",
    r"\bbê tê ét\b": "BTS",
    r"\bVê rờ ét\b": "VRS",
    r"\bxê ét ích\b": "CSX",
    # Khác
    r"\bêy ai\b": "AI",
}


def clean_ssml_tags(text: str) -> str:
    """Xóa tất cả SSML tags (VD: <break time='0.5s' />)"""
    return re.sub(r'<break[^>]+/?>', '', text)


def clean_emotion_tags(text: str) -> str:
    """Xóa emotion tags (VD: [thoughtful], [surprised])"""
    return re.sub(r'\[\w+\]', '', text)


def normalize_phonetics(
    text: str,
    phonetic_map: Optional[Dict[str, str]] = None,
) -> str:
    """Chuẩn hóa từ vựng phát âm → chữ viết tắt."""
    if phonetic_map is None:
        phonetic_map = PHONETIC_MAP
    for pattern, replacement in phonetic_map.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def normalize_whitespace(text: str) -> str:
    """Xóa khoảng trắng thừa và ký tự đặc biệt dư."""
    text = text.replace('—', '')
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_script(
    text: str,
    custom_phonetic_map: Optional[Dict[str, str]] = None,
) -> str:
    """
    Pipeline làm sạch đầy đủ cho raw script từ HeyGen.

    Steps:
        1. Xóa SSML tags (<break>, etc.)
        2. Xóa emotion tags ([thoughtful], [surprised], etc.)
        3. Chuẩn hóa từ vựng (phát âm → viết tắt)
        4. Xóa ký tự/khoảng trắng thừa
    """
    text = clean_ssml_tags(text)
    text = clean_emotion_tags(text)
    text = normalize_phonetics(text, custom_phonetic_map)
    text = normalize_whitespace(text)
    return text


def clean_text(text: str) -> str:
    """
    Làm sạch nhẹ cho bất kỳ text nào (SRT segments, Whisper output, etc.).
    Chỉ chuẩn hóa từ vựng + whitespace, KHÔNG xóa SSML/emotion tags
    (vì Whisper output không có các tags đó).
    """
    text = normalize_phonetics(text)
    text = normalize_whitespace(text)
    return text


def clean_segments(
    segments: List[Dict],
) -> List[Dict]:
    """
    Áp dụng clean_text cho từng segment trong danh sách.
    Dùng cho Whisper segments hoặc parsed SRT cues trước khi export KB.

    Args:
        segments: List of dicts, mỗi dict có key 'text'.

    Returns:
        List mới với text đã được clean.
    """
    cleaned = []
    for seg in segments:
        new_seg = dict(seg)
        new_seg["text"] = clean_text(seg["text"])
        cleaned.append(new_seg)
    return cleaned


def split_into_sentences(text: str) -> List[str]:
    """Tách văn bản đã clean thành các câu dựa trên dấu câu."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]
