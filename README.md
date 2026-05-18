# Viettel AI Fresher — Interactive Learning Platform

Hệ thống học liệu AI tích hợp Avatar tương tác thời gian thực (Anam AI),  
video bài giảng (HeyGen), và pipeline xử lý script tự động.

## 📁 Cấu trúc dự án

```
Anam_Viettel_AI_Fresher/
│
├── src/                          # Source code
│   ├── server/                   # FastAPI web server
│   │   └── app.py                # Entry point server
│   │
│   └── pipeline/                 # Pipeline xử lý script bài giảng
│       ├── cleaner.py            # Làm sạch SSML, emotion tags, chuẩn hóa từ
│       ├── transcriber.py        # Whisper STT — video → segments
│       ├── translator.py         # Google Translate — dịch đa ngôn ngữ
│       ├── srt_formatter.py      # SRT format utilities
│       ├── kb_exporter.py        # Export + upload Anam Knowledge Base
│       └── run_pipeline.py       # CLI entry point
│
├── web/                          # Giao diện web (HTML/CSS/JS)
│   └── index.html
│
├── data/                         # Input & output data
│   ├── videos/                   # Video bài giảng (input)
│   ├── scripts/                  # Raw scripts từ HeyGen (input)
│   └── output/                   # Kết quả pipeline
│       ├── srt/                  # File .srt (VI, EN, bilingual)
│       └── kb/                   # File KB cho Anam (.txt, .json)
│
├── assets/                       # Static assets (logo, images)
├── .env                          # API keys (không commit)
├── requirements.txt              # Python dependencies
└── README.md
```

## 🚀 Cài đặt và Khởi động

### Yêu cầu tiên quyết
- Python 3.9+
- **[FFmpeg](https://ffmpeg.org/download.html)** (bắt buộc cho Whisper)

### Bước 1: Tạo môi trường ảo
```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

> Nếu Windows báo lỗi `Execution_Policies`:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
> ```

### Bước 2: Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### Bước 3: Cấu hình API keys
Tạo file `.env` ở thư mục gốc:
```env
ANAM_API_KEY=your_anam_api_key
ANAM_PERSONA_ID=your_anam_persona_id
HEYGEN_API_KEY=your_heygen_api_key
```

### Bước 4: Khởi động Server
```bash
python -m src.server.app
```
Truy cập: **[http://localhost:8000](http://localhost:8000)**

---

## 🔧 Pipeline xử lý Script

Pipeline tự động: **Raw script → Clean → Translate → SRT → KB export → Upload Anam**

### Chạy toàn bộ pipeline
```bash
# Chỉ clean script
python -m src.pipeline.run_pipeline \
    --script data/scripts/scripts.txt \
    --title "Lịch sử Viettel - Giai đoạn 3"

# Full pipeline (video + script)
python -m src.pipeline.run_pipeline \
    --video data/videos/bai_giang.mp4 \
    --script data/scripts/scripts.txt \
    --title "Lịch sử Viettel - Giai đoạn 3"

# Pipeline + upload lên Anam KB
python -m src.pipeline.run_pipeline \
    --video data/videos/bai_giang.mp4 \
    --upload-kb --kb-folder-id <FOLDER_UUID>
```

### Pipeline modules

| Module | Chức năng |
|--------|-----------|
| `cleaner.py` | Xóa SSML/emotion tags, chuẩn hóa phát âm → viết tắt |
| `transcriber.py` | Whisper STT — trích xuất audio → segments có timestamp |
| `translator.py` | Google Translate — dịch VI ↔ EN |
| `srt_formatter.py` | Tạo/parse file SRT (đơn ngữ, song ngữ) |
| `kb_exporter.py` | Export TXT/JSON cho Anam KB + upload qua API |

### Output files
```
data/output/
├── srt/
│   ├── subtitle_vi.srt        # Phụ đề tiếng Việt
│   ├── subtitle_en.srt        # Phụ đề tiếng Anh
│   └── subtitle_vi_en.srt     # Phụ đề song ngữ
├── kb/
│   ├── *_kb.txt               # Knowledge Base (text)
│   └── *_kb.json              # Knowledge Base (JSON)
├── scripts_clean_vi.txt        # Script đã làm sạch
└── scripts_bilingual.txt       # Script song ngữ
```

---

## 📡 Tính năng Web Platform

- 🎬 **Video bài giảng** — Phát video với phụ đề song ngữ đồng bộ
- 🤖 **Live Avatar** — Tương tác hỏi đáp với giảng viên ảo (Anam AI)
- 🎤 **Voice Input** — Nhận diện giọng nói tiếng Việt (Whisper STT)
- 📋 **HeyGen Integration** — Duyệt video bài giảng từ HeyGen
- 💬 **Chat** — Lịch sử hội thoại real-time

---

*Powered by Anam AI, HeyGen, OpenAI Whisper*