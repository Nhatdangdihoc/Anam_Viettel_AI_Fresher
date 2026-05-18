"""
SRT Formatter Module
Utilities cho định dạng file phụ đề SRT (SubRip Subtitle).
"""

from typing import List, Dict, Any


# Type alias
Segment = Dict[str, Any]


def format_srt_time(seconds: float) -> str:
    """Chuyển số giây → định dạng SRT: HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def parse_srt_time(time_str: str) -> float:
    """Parse SRT time string → seconds."""
    hms, ms = time_str.strip().split(",")
    h, m, s = hms.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def segments_to_srt(segments: List[Segment]) -> str:
    """Chuyển Whisper segments → chuỗi SRT đơn ngữ."""
    lines = []
    for i, seg in enumerate(segments, 1):
        start = format_srt_time(seg["start"])
        end = format_srt_time(seg["end"])
        text = seg["text"].strip()
        lines.append(f"{i}")
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def segments_to_bilingual_srt(
    segments: List[Segment],
    translated_texts: List[str],
) -> str:
    """Tạo SRT song ngữ: dòng 1 = VI, dòng 2 = EN."""
    lines = []
    for i, (seg, en_text) in enumerate(zip(segments, translated_texts), 1):
        start = format_srt_time(seg["start"])
        end = format_srt_time(seg["end"])
        vi_text = seg["text"].strip()
        lines.append(f"{i}")
        lines.append(f"{start} --> {end}")
        lines.append(vi_text)
        lines.append(en_text)
        lines.append("")
    return "\n".join(lines)


def parse_srt(srt_text: str) -> List[Dict[str, Any]]:
    """
    Parse file SRT → list of cues.

    Returns:
        [{index, start, end, text}, ...]
    """
    blocks = srt_text.strip().split("\n\n")
    cues = []
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        try:
            index = int(lines[0])
        except ValueError:
            continue
        time_parts = lines[1].split("-->")
        if len(time_parts) != 2:
            continue
        start = parse_srt_time(time_parts[0])
        end = parse_srt_time(time_parts[1])
        text = "\n".join(lines[2:])
        cues.append({"index": index, "start": start, "end": end, "text": text})
    return cues


def create_bilingual_text(
    vi_sentences: List[str],
    en_sentences: List[str],
) -> str:
    """Tạo format kịch bản song ngữ dạng text."""
    lines = []
    for i, (vi, en) in enumerate(zip(vi_sentences, en_sentences), 1):
        lines.append(f"Subtitle {i}:")
        lines.append(f"VI: {vi}")
        lines.append(f"EN: {en}")
        lines.append("")
    return "\n".join(lines)
