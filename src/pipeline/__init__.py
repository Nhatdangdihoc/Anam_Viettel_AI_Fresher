"""
Pipeline xử lý script bài giảng cho Anam AI Knowledge Base.

Modules:
    - cleaner:       Làm sạch SSML tags, emotion tags, chuẩn hóa từ vựng
    - transcriber:   Whisper STT — trích xuất audio từ video → segments
    - translator:    Google Translate — dịch thuật đa ngôn ngữ
    - srt_formatter: Utilities cho định dạng SRT
    - kb_exporter:   Export + upload Knowledge Base files cho Anam AI
"""

__version__ = "1.0.0"
