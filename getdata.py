import os
import time
import sounddevice as sd
from scipy.io.wavfile import write
import pandas as pd
import google.generativeai as genai

# ======== CẤU HÌNH GEMINI ========
GOOGLE_API_KEY = "AIzaSyAOpTOA26r2zfRpU-5SL430Elu0pY7B3js"  # <-- Thay bằng API Key thật của bạn
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

# ======== HÀM SINH VĂN BẢN =========
def generate_vietnamese_text():
    prompt = "Viết một đoạn văn bằng tiếng Việt dài khoảng 80 đến 120 từ về một chủ đề bất kỳ bao gồm cả những từ khó và những chủ đề đặc biệt hơn. (ví dụ: công nghệ, văn hoá, lịch sử, game, v.v.)"
    response = model.generate_content(prompt)
    return response.text.strip()

# ======== HÀM THU ÂM =========
def record_audio(filename, duration=40, fs=16000):
    print("🗣️ Bắt đầu thu âm trong 3 giây...")
    time.sleep(3)
    print("🎙️ Đang thu âm...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    print("✅ Thu âm hoàn tất.")
    write(filename, fs, recording)

# ======== CHẠY CHƯƠNG TRÌNH =========
def main():
    speaker = input("Nhập tên người thu âm: ").strip()
    os.makedirs("recordings", exist_ok=True)

    csv_file = "speech_data.csv"
    all_data = []

    for i in range(3):
        print(f"\n--- Lần {i+1}/3 ---")

        # Sinh văn bản
        text = generate_vietnamese_text()
        print("Đoạn văn cần đọc:")
        print(text)

        # Thu âm và lưu file
        filename = f"recordings/{speaker}_sample_{int(time.time())}.wav"
        record_audio(filename)

        # Ghi dữ liệu
        all_data.append({
            "file_path": filename,
            "label": text,
            "speaker": speaker
        })

    # Ghi vào CSV
    df_new = pd.DataFrame(all_data)
    if os.path.exists(csv_file):
        df_old = pd.read_csv(csv_file)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_csv(csv_file, index=False, encoding="utf-8-sig")
    print(f"\n📁 Dữ liệu đã được lưu vào {csv_file}")

if __name__ == "__main__":
    main()
