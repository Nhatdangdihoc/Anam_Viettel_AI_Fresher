"""
Knowledge Base Exporter
Export dữ liệu bài giảng sang file chuẩn cho Anam AI Knowledge Base.
Hỗ trợ upload trực tiếp lên Anam KB qua REST API.

Anam KB chấp nhận: TXT, MD, JSON, PDF, DOCX, CSV, LOG
→ Pipeline convert SRT/script → TXT hoặc JSON trước khi upload.
"""

import json
import os
import re
from datetime import datetime
from typing import List, Dict, Any, Optional


# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def _format_timestamp(seconds: float) -> str:
    """Format seconds → MM:SS cho KB readability."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def _safe_filename(title: str) -> str:
    """Tạo tên file an toàn từ tiêu đề."""
    safe = re.sub(r'[^\w\s\-]', '', title)
    safe = re.sub(r'\s+', '_', safe).strip('_')
    return safe[:80] if safe else "lecture"


def srt_to_kb_text(
    srt_cues: List[Dict[str, Any]],
    lecture_title: str = "Bài giảng",
    include_timestamps: bool = True,
) -> str:
    """
    Chuyển SRT cues → văn bản có cấu trúc cho Anam Knowledge Base.

    Format output:
        # Bài giảng: Lịch sử phát triển Viettel
        [00:00] Xin chào các bạn...
        [00:05] Nội dung tiếp theo...
    """
    lines = [
        f"# {lecture_title}",
        f"# Xuất bởi Viettel AI Fresher Pipeline — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"# Tổng số phân đoạn: {len(srt_cues)}",
        "",
    ]

    for cue in srt_cues:
        text = cue["text"].strip()
        if not text:
            continue
        if include_timestamps:
            ts = _format_timestamp(cue["start"])
            lines.append(f"[{ts}] {text}")
        else:
            lines.append(text)

    return "\n".join(lines)


def srt_to_kb_json(
    srt_cues: List[Dict[str, Any]],
    lecture_title: str = "Bài giảng",
    metadata: Optional[Dict] = None,
) -> str:
    """
    Chuyển SRT cues → JSON cho Anam Knowledge Base.

    Output JSON structure:
        {
            "title": "...",
            "exported_at": "...",
            "segments": [{start, end, text}, ...]
        }
    """
    doc = {
        "title": lecture_title,
        "exported_at": datetime.now().isoformat(),
        "total_segments": len(srt_cues),
        "metadata": metadata or {},
        "segments": [
            {
                "index": cue.get("index", i + 1),
                "start": round(cue["start"], 3),
                "end": round(cue["end"], 3),
                "text": cue["text"].strip(),
            }
            for i, cue in enumerate(srt_cues)
        ],
    }
    return json.dumps(doc, ensure_ascii=False, indent=2)


def clean_script_to_kb_text(
    cleaned_text: str,
    lecture_title: str = "Bài giảng",
) -> str:
    """
    Chuyển script đã clean → văn bản KB (không có timestamp).
    Phù hợp khi chỉ có script text mà không có video.
    """
    lines = [
        f"# {lecture_title}",
        f"# Xuất bởi Viettel AI Fresher Pipeline — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        cleaned_text,
    ]
    return "\n".join(lines)


def export_kb_files(
    output_dir: str,
    lecture_title: str = "Bài giảng",
    srt_cues: Optional[List[Dict[str, Any]]] = None,
    cleaned_text: Optional[str] = None,
    formats: Optional[List[str]] = None,
) -> List[str]:
    """
    Export KB files ra đĩa.

    Args:
        output_dir:    Thư mục output.
        lecture_title:  Tiêu đề bài giảng.
        srt_cues:      Parsed SRT cues (nếu có).
        cleaned_text:  Script đã clean (nếu có).
        formats:       Danh sách format cần export ('txt', 'json').

    Returns:
        Danh sách đường dẫn file đã export.
    """
    if formats is None:
        formats = ["txt", "json"]

    os.makedirs(output_dir, exist_ok=True)
    exported = []
    safe_name = _safe_filename(lecture_title)

    if "txt" in formats:
        if srt_cues:
            content = srt_to_kb_text(srt_cues, lecture_title)
        elif cleaned_text:
            content = clean_script_to_kb_text(cleaned_text, lecture_title)
        else:
            print("  [KB] Không có dữ liệu để export TXT")
            content = None

        if content:
            path = os.path.join(output_dir, f"{safe_name}_kb.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            exported.append(path)
            print(f"  [KB] Exported TXT: {path}")

    if "json" in formats and srt_cues:
        content = srt_to_kb_json(srt_cues, lecture_title)
        path = os.path.join(output_dir, f"{safe_name}_kb.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        exported.append(path)
        print(f"  [KB] Exported JSON: {path}")

    return exported


# ============================================================================
# ANAM KB UPLOAD
# ============================================================================

async def upload_to_anam_kb(
    file_path: str,
    api_key: str,
    folder_id: str,
    base_url: str = "https://api.anam.ai",
) -> Dict[str, Any]:
    """
    Upload file lên Anam Knowledge Base qua REST API.

    Endpoint: POST /v1/knowledge/groups/{folder_id}/documents
    Supported: TXT, MD, JSON, PDF, DOCX, CSV, LOG (max 50MB)

    Args:
        file_path:  Đường dẫn file cần upload.
        api_key:    Anam API key.
        folder_id:  ID của knowledge folder trên Anam.
        base_url:   Anam API base URL.

    Returns:
        Response data từ Anam API.
    """
    import httpx

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File không tồn tại: {file_path}")

    filename = os.path.basename(file_path)
    ext = os.path.splitext(filename)[1].lower()

    mime_map = {
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".json": "application/json",
        ".pdf": "application/pdf",
        ".csv": "text/csv",
        ".log": "text/plain",
    }
    mime_type = mime_map.get(ext, "text/plain")

    url = f"{base_url}/v1/knowledge/groups/{folder_id}/documents"

    async with httpx.AsyncClient(timeout=60.0) as client:
        with open(file_path, "rb") as f:
            files = {"file": (filename, f, mime_type)}
            headers = {"Authorization": f"Bearer {api_key}"}

            print(f"  [KB] Uploading '{filename}' to Anam KB folder {folder_id}...")
            resp = await client.post(url, headers=headers, files=files)
            resp.raise_for_status()

            result = resp.json()
            print(f"  [KB] Upload thành công! Status: {result.get('status', 'OK')}")
            return result


def upload_to_anam_kb_sync(
    file_path: str,
    api_key: str,
    folder_id: str,
    base_url: str = "https://api.anam.ai",
) -> Dict[str, Any]:
    """
    Phiên bản synchronous của upload_to_anam_kb.
    Dùng cho CLI (run_pipeline.py).
    """
    import httpx

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File không tồn tại: {file_path}")

    filename = os.path.basename(file_path)
    ext = os.path.splitext(filename)[1].lower()

    mime_map = {
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".json": "application/json",
        ".pdf": "application/pdf",
        ".csv": "text/csv",
        ".log": "text/plain",
    }
    mime_type = mime_map.get(ext, "text/plain")

    url = f"{base_url}/v1/knowledge/groups/{folder_id}/documents"

    with httpx.Client(timeout=60.0) as client:
        with open(file_path, "rb") as f:
            files = {"file": (filename, f, mime_type)}
            headers = {"Authorization": f"Bearer {api_key}"}

            print(f"  [KB] Uploading '{filename}' to Anam KB folder {folder_id}...")
            resp = client.post(url, headers=headers, files=files)
            resp.raise_for_status()

            result = resp.json()
            print(f"  [KB] Upload thành công! Status: {result.get('status', 'OK')}")
            return result
