"""
Pipeline Runner CLI
Chạy toàn bộ pipeline: Script → Clean → Translate → SRT → KB Export → (Upload)

Usage:
    python -m src.pipeline.run_pipeline --help
    python -m src.pipeline.run_pipeline --script data/scripts/scripts.txt --output data/output
    python -m src.pipeline.run_pipeline --video data/videos/bai_giang.mp4 --output data/output
    python -m src.pipeline.run_pipeline --video data/videos/bai_giang.mp4 --upload-kb --kb-folder-id <ID>
"""

import argparse
import os
import sys

# Fix Windows console encoding
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.pipeline.cleaner import clean_script, clean_segments, split_into_sentences
from src.pipeline.translator import translate_sentences
from src.pipeline.srt_formatter import (
    segments_to_srt,
    segments_to_bilingual_srt,
    parse_srt,
    create_bilingual_text,
)
from src.pipeline.kb_exporter import export_kb_files, upload_to_anam_kb_sync


def run(args: argparse.Namespace) -> None:
    """Chạy pipeline chính."""
    print("=" * 60)
    print("  VIETTEL AI FRESHER — SCRIPT PROCESSING PIPELINE")
    print("=" * 60)

    os.makedirs(args.output, exist_ok=True)
    srt_dir = os.path.join(args.output, "srt")
    kb_dir = os.path.join(args.output, "kb")
    os.makedirs(srt_dir, exist_ok=True)
    os.makedirs(kb_dir, exist_ok=True)

    segments = None
    cleaned_text = None

    # ── PART A: Script text cleaning & translation ────────────────
    if args.script and os.path.exists(args.script):
        print(f"\n[PART A] Script text cleaning & translation")
        print(f"  Input: {args.script}")

        with open(args.script, "r", encoding="utf-8") as f:
            raw_text = f.read()

        print("  [1/4] Cleaning script...")
        cleaned_text = clean_script(raw_text)
        clean_path = os.path.join(args.output, "scripts_clean_vi.txt")
        with open(clean_path, "w", encoding="utf-8") as f:
            f.write(cleaned_text)
        print(f"        → {clean_path}")

        print("  [2/4] Splitting into sentences...")
        vi_sentences = split_into_sentences(cleaned_text)
        print(f"        → {len(vi_sentences)} sentences")

        if not args.skip_translate:
            print("  [3/4] Translating to English...")
            en_sentences = translate_sentences(vi_sentences)

            print("  [4/4] Creating bilingual text...")
            bilingual = create_bilingual_text(vi_sentences, en_sentences)
            bilingual_path = os.path.join(args.output, "scripts_bilingual.txt")
            with open(bilingual_path, "w", encoding="utf-8") as f:
                f.write(bilingual)
            print(f"        → {bilingual_path}")
        else:
            print("  [3/4] Skipping translation (--skip-translate)")

    elif args.script:
        print(f"\n  [!] Script file not found: {args.script}")

    # ── PART B: Whisper video → SRT with timestamps ───────────────
    if args.video and os.path.exists(args.video):
        print(f"\n[PART B] Whisper SRT generation")
        print(f"  Input: {args.video}")

        from src.pipeline.transcriber import transcribe_video

        print(f"  [1/4] Transcribing ({args.language}, model={args.whisper_model})...")
        segments = transcribe_video(
            args.video,
            language=args.language,
            model_name=args.whisper_model,
        )

        # SRT tiếng Việt
        print("  [2/4] Writing Vietnamese SRT...")
        srt_vi = segments_to_srt(segments)
        srt_vi_path = os.path.join(srt_dir, "subtitle_vi.srt")
        with open(srt_vi_path, "w", encoding="utf-8") as f:
            f.write(srt_vi)
        print(f"        → {srt_vi_path}")

        if not args.skip_translate:
            # Dịch segments → tiếng Anh
            print("  [3/4] Translating segments to English...")
            vi_texts = [seg["text"].strip() for seg in segments]
            en_texts = translate_sentences(vi_texts)

            # SRT tiếng Anh
            srt_en_lines = []
            for i, (seg, en) in enumerate(zip(segments, en_texts), 1):
                from src.pipeline.srt_formatter import format_srt_time
                start = format_srt_time(seg["start"])
                end = format_srt_time(seg["end"])
                srt_en_lines.extend([f"{i}", f"{start} --> {end}", en, ""])
            srt_en_path = os.path.join(srt_dir, "subtitle_en.srt")
            with open(srt_en_path, "w", encoding="utf-8") as f:
                f.write("\n".join(srt_en_lines))
            print(f"        → {srt_en_path}")

            # SRT song ngữ
            print("  [4/4] Writing bilingual SRT...")
            srt_dual = segments_to_bilingual_srt(segments, en_texts)
            srt_dual_path = os.path.join(srt_dir, "subtitle_vi_en.srt")
            with open(srt_dual_path, "w", encoding="utf-8") as f:
                f.write(srt_dual)
            print(f"        → {srt_dual_path}")
        else:
            print("  [3/4] Skipping translation (--skip-translate)")

    elif args.video:
        print(f"\n  [!] Video file not found: {args.video}")

    # ── PART C: Export Knowledge Base files ────────────────────────
    print(f"\n[PART C] Knowledge Base export")

    srt_cues = None
    if segments:
        # Convert Whisper segments → cues format cho KB
        srt_cues = [
            {"index": i + 1, "start": s["start"], "end": s["end"], "text": s["text"]}
            for i, s in enumerate(segments)
        ]
    elif os.path.exists(os.path.join(srt_dir, "subtitle_vi.srt")):
        # Fallback: đọc file SRT đã có sẵn
        print("  [INFO] Using existing SRT file for KB export")
        with open(os.path.join(srt_dir, "subtitle_vi.srt"), "r", encoding="utf-8") as f:
            srt_cues = parse_srt(f.read())

    # Áp dụng cleaner cho SRT cues (chuẩn hóa "Việt Theo" → "Viettel", etc.)
    if srt_cues:
        print("  [CLEAN] Cleaning SRT segments (phonetic normalization)...")
        srt_cues = clean_segments(srt_cues)

    exported = export_kb_files(
        output_dir=kb_dir,
        lecture_title=args.title,
        srt_cues=srt_cues,
        cleaned_text=cleaned_text,
        formats=["txt", "json"],
    )

    # ── PART D: Upload to Anam KB (optional) ──────────────────────
    if args.upload_kb:
        print(f"\n[PART D] Upload to Anam Knowledge Base")

        if not args.kb_folder_id:
            print("  [!] Cần --kb-folder-id để upload. Bỏ qua upload.")
        else:
            api_key = args.anam_api_key or os.getenv("ANAM_API_KEY", "")
            if not api_key:
                print("  [!] Cần ANAM_API_KEY (--anam-api-key hoặc .env). Bỏ qua upload.")
            else:
                for fpath in exported:
                    try:
                        upload_to_anam_kb_sync(
                            file_path=fpath,
                            api_key=api_key,
                            folder_id=args.kb_folder_id,
                        )
                    except Exception as e:
                        print(f"  [!] Upload failed for {fpath}: {e}")

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  DONE! Output files:")
    for root, dirs, files in os.walk(args.output):
        for fname in files:
            fpath = os.path.join(root, fname)
            size = os.path.getsize(fpath)
            print(f"    [OK] {fpath} ({size:,} bytes)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Viettel AI Fresher — Script Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Chỉ clean script:
  python -m src.pipeline.run_pipeline --script data/scripts/scripts.txt

  # Full pipeline (video + script):
  python -m src.pipeline.run_pipeline \\
      --video data/videos/bai_giang.mp4 \\
      --script data/scripts/scripts.txt \\
      --title "Lịch sử Viettel - Giai đoạn 3"

  # Pipeline + upload KB:
  python -m src.pipeline.run_pipeline \\
      --video data/videos/bai_giang.mp4 \\
      --upload-kb --kb-folder-id <FOLDER_UUID>
        """,
    )

    parser.add_argument(
        "--video", type=str, default=None,
        help="Đường dẫn video bài giảng (input cho Whisper)",
    )
    parser.add_argument(
        "--script", type=str, default=None,
        help="Đường dẫn raw script text (input cho cleaner)",
    )
    parser.add_argument(
        "--output", type=str, default="data/output",
        help="Thư mục output (default: data/output)",
    )
    parser.add_argument(
        "--title", type=str,
        default="Bài giảng Viettel AI",
        help="Tiêu đề bài giảng cho KB export",
    )
    parser.add_argument(
        "--whisper-model", type=str, default="small",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model (default: small)",
    )
    parser.add_argument(
        "--language", type=str, default="vi",
        help="Ngôn ngữ nguồn (default: vi)",
    )
    parser.add_argument(
        "--skip-translate", action="store_true",
        help="Bỏ qua bước dịch thuật",
    )
    parser.add_argument(
        "--upload-kb", action="store_true",
        help="Upload KB files lên Anam sau khi export",
    )
    parser.add_argument(
        "--kb-folder-id", type=str, default=None,
        help="Anam KB folder ID (UUID) cho upload",
    )
    parser.add_argument(
        "--anam-api-key", type=str, default=None,
        help="Anam API key (hoặc dùng biến môi trường ANAM_API_KEY)",
    )

    args = parser.parse_args()

    if not args.video and not args.script:
        parser.error("Cần ít nhất --video hoặc --script để chạy pipeline.")

    run(args)


if __name__ == "__main__":
    main()
