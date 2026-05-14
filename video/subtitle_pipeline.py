import re
import os
import sys
import time

# Fix Windows console encoding for Vietnamese
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ============================================================================
# 1. TU DIEN CHUAN HOA PHAT AM -> CHU VIET TAT
# ============================================================================
PHONETIC_MAP = {
    r"\b4 gờ\b": "4G",
    r"\b3 gờ\b": "3G",
    r"\bbê tê ét\b": "BTS",
    r"\bVê rờ ét\b": "VRS",
    r"\bxê ét ích\b": "CSX",
    r"\bViệt theo\b": "Viettel",
    r"\bêy ai\b": "AI"
}


def clean_script(text):
    """Làm sạch các thẻ SSML, Tag cảm xúc và chuẩn hóa từ vựng"""
    # Xóa thẻ break (VD: <break time="0.5s" />)
    text = re.sub(r'<break[^>]+/>', '', text)

    # Xóa thẻ cảm xúc (VD: [thoughtful], [surprised])
    text = re.sub(r'\[\w+\]', '', text)

    # Xóa ký tự gạch ngang dư thừa
    text = text.replace('—', '')

    # Chuẩn hóa từ vựng (phát âm -> chữ viết)
    for pattern, replacement in PHONETIC_MAP.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Xóa khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def split_into_sentences(text):
    """Tách văn bản thành các câu để làm phụ đề"""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


# ============================================================================
# 2. TRANSLATION ENGINE (Google Translate via deep_translator)
# ============================================================================
from deep_translator import GoogleTranslator


def translate_sentences(sentences, src='vi', dest='en'):
    """Dịch danh sách câu sang ngôn ngữ đích."""
    translator = GoogleTranslator(source=src, target=dest)
    translated = []

    for i, s in enumerate(sentences):
        try:
            result = translator.translate(s)
            translated.append(result if result else s)
            time.sleep(0.3)
        except Exception as e:
            print(f"  [!] Translation error at sentence {i}: {e}")
            translated.append(f"[Translation Error] {s}")

    return translated


# ============================================================================
# 3. BILINGUAL TEXT FORMAT (VI/EN side by side)
# ============================================================================
def create_bilingual_text(vi_sentences, en_sentences):
    """Tạo format kịch bản song ngữ dạng text."""
    lines = []
    for i, (vi, en) in enumerate(zip(vi_sentences, en_sentences), 1):
        lines.append(f"Subtitle {i}:")
        lines.append(f"VI: {vi}")
        lines.append(f"EN: {en}")
        lines.append("")
    return "\n".join(lines)


# ============================================================================
# 4. WHISPER SRT GENERATOR (tạo .srt có timestamp từ video/audio)
# ============================================================================
def generate_srt_from_video(video_path, language='vi', model_name='small'):
    """
    Dùng Whisper để trích xuất audio từ video,
    tạo file .srt (phụ đề tiếng Việt) có timestamp chính xác.
    Trả về danh sách segments [{start, end, text}, ...]
    """
    import whisper

    print(f"  Loading Whisper model '{model_name}'...")
    model = whisper.load_model(model_name)

    print(f"  Transcribing '{os.path.basename(video_path)}'...")
    result = model.transcribe(video_path, language=language, fp16=False)

    return result.get("segments", [])


def format_srt_time(seconds):
    """Chuyển số giây thành định dạng SRT: HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def segments_to_srt(segments):
    """Chuyển danh sách segments thành chuỗi SRT."""
    srt_lines = []
    for i, seg in enumerate(segments, 1):
        start = format_srt_time(seg["start"])
        end = format_srt_time(seg["end"])
        text = seg["text"].strip()
        srt_lines.append(f"{i}")
        srt_lines.append(f"{start} --> {end}")
        srt_lines.append(text)
        srt_lines.append("")
    return "\n".join(srt_lines)


def segments_to_bilingual_srt(segments, translated_texts):
    """Tạo file SRT song ngữ: dòng 1 = VI, dòng 2 = EN."""
    srt_lines = []
    for i, (seg, en_text) in enumerate(zip(segments, translated_texts), 1):
        start = format_srt_time(seg["start"])
        end = format_srt_time(seg["end"])
        vi_text = seg["text"].strip()
        srt_lines.append(f"{i}")
        srt_lines.append(f"{start} --> {end}")
        srt_lines.append(vi_text)
        srt_lines.append(en_text)
        srt_lines.append("")
    return "\n".join(srt_lines)


# ============================================================================
# 5. MAIN PIPELINE
# ============================================================================
if __name__ == "__main__":
    script_path = "scripts.txt"
    video_path = "Lịch sử phát triển Viettel - Giai đoạn 3 (2010-2020)_1080p.mp4"

    # Output files
    out_clean_vi    = "scripts_clean_vi.txt"
    out_bilingual   = "scripts_bilingual.txt"
    out_srt_vi      = "subtitle_vi.srt"
    out_srt_en      = "subtitle_en.srt"
    out_srt_dual    = "subtitle_vi_en.srt"

    print("=" * 60)
    print("  SUBTITLE PIPELINE - Viettel AI Fresher")
    print("=" * 60)

    # ── PART A: Clean script text + translate ──────────────────
    print("\n[PART A] Script text cleaning & translation")

    if os.path.exists(script_path):
        with open(script_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        print("  [1/4] Cleaning script...")
        cleaned_text = clean_script(raw_text)
        with open(out_clean_vi, "w", encoding="utf-8") as f:
            f.write(cleaned_text)
        print(f"        -> {out_clean_vi}")

        print("  [2/4] Splitting into sentences...")
        vi_sentences = split_into_sentences(cleaned_text)
        print(f"        -> {len(vi_sentences)} sentences")

        print("  [3/4] Translating to English (Google Translate)...")
        en_sentences = translate_sentences(vi_sentences)

        print("  [4/4] Creating bilingual text...")
        bilingual = create_bilingual_text(vi_sentences, en_sentences)
        with open(out_bilingual, "w", encoding="utf-8") as f:
            f.write(bilingual)
        print(f"        -> {out_bilingual}")
    else:
        print(f"  [!] Script file not found: {script_path}")

    # ── PART B: Whisper video -> SRT with timestamps ───────────
    print(f"\n[PART B] Whisper SRT generation from video")

    if os.path.exists(video_path):
        print("  [1/4] Running Whisper transcription (Vietnamese)...")
        segments = generate_srt_from_video(video_path, language='vi', model_name='small')
        print(f"        -> {len(segments)} segments detected")

        # SRT tiếng Việt
        print("  [2/4] Writing Vietnamese SRT...")
        srt_vi = segments_to_srt(segments)
        with open(out_srt_vi, "w", encoding="utf-8") as f:
            f.write(srt_vi)
        print(f"        -> {out_srt_vi}")

        # Dịch từng segment sang tiếng Anh
        print("  [3/4] Translating segments to English...")
        vi_texts = [seg["text"].strip() for seg in segments]
        en_texts = translate_sentences(vi_texts)

        # SRT tiếng Anh
        srt_en_lines = []
        for i, (seg, en) in enumerate(zip(segments, en_texts), 1):
            start = format_srt_time(seg["start"])
            end = format_srt_time(seg["end"])
            srt_en_lines.append(f"{i}")
            srt_en_lines.append(f"{start} --> {end}")
            srt_en_lines.append(en)
            srt_en_lines.append("")
        with open(out_srt_en, "w", encoding="utf-8") as f:
            f.write("\n".join(srt_en_lines))
        print(f"        -> {out_srt_en}")

        # SRT song ngữ
        print("  [4/4] Writing bilingual SRT (VI + EN)...")
        srt_dual = segments_to_bilingual_srt(segments, en_texts)
        with open(out_srt_dual, "w", encoding="utf-8") as f:
            f.write(srt_dual)
        print(f"        -> {out_srt_dual}")

    else:
        print(f"  [!] Video file not found: {video_path}")

    print("\n" + "=" * 60)
    print("  DONE! All output files:")
    for f in [out_clean_vi, out_bilingual, out_srt_vi, out_srt_en, out_srt_dual]:
        exists = "OK" if os.path.exists(f) else "SKIP"
        print(f"    [{exists}] {f}")
    print("=" * 60)
