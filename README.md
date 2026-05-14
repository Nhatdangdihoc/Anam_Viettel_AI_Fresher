# DU_AN_ANAM

Dự án tương tác Avatar AI (HeyGen/Anam) tích hợp nhận diện giọng nói (Whisper).

## 🚀 Cài đặt và Khởi động

### Yêu cầu tiên quyết
- Python 3.9 trở lên.
- **[FFmpeg](https://ffmpeg.org/download.html)** (Bắt buộc cần cài đặt trong hệ thống để mô hình Whisper xử lý nhận diện giọng nói).

### Bước 1: Mở Terminal
Mở VS Code tại thư mục dự án và mở Terminal (Nhấn `Ctrl + \`` hoặc vào `View → Terminal`).

### Bước 2: Tạo và kích hoạt môi trường ảo
```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường ảo (Trên Windows)
.\venv\Scripts\activate
```

*(Lưu ý: Nếu Windows báo lỗi `Execution_Policies`, hãy chạy lệnh sau để khắc phục, sau đó chạy lại lệnh kích hoạt ở trên:*
```powershell
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
```
*)*

### Bước 3: Cài đặt các thư viện cần thiết
```bash
pip install -r requirements.txt
```

### Bước 4: Cấu hình biến môi trường
Tạo một file `.env` ở thư mục gốc của dự án (hoặc trong thư mục `Backend`) và bổ sung các thông tin API sau:
```env
ANAM_API_KEY=your_anam_api_key_here
ANAM_PERSONA_ID=your_anam_persona_id_here
HEYGEN_API_KEY=your_heygen_api_key_here
```

### Bước 5: Khởi động Server
Chuyển vào thư mục `Backend` và chạy file `main.py`:
```bash
cd Backend
python main.py
```

### Bước 6: Truy cập Ứng dụng
Mở trình duyệt và truy cập vào địa chỉ hiển thị trong Terminal, mặc định là: **[http://localhost:8000](http://localhost:8000)**.