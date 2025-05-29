import google.generativeai as genai
import pandas as pd
import os
import json
from datetime import datetime
from typing import List, Dict
import random
import sounddevice as sd
import soundfile as sf
import numpy as np
import threading
import time
import io

class VoiceDatasetGenerator:
    def __init__(self, gemini_api_key: str, output_dir: str = "voice_dataset", 
                 sample_rate: int = 22050, duration: int = 10):
        """
        Khởi tạo generator cho voice dataset
        
        Args:
            gemini_api_key: API key của Google Gemini
            output_dir: Thư mục lưu dữ liệu
            sample_rate: Tần số mẫu (Hz)
            duration: Thời gian thu tối đa (giây)
        """
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        self.output_dir = output_dir
        self.audio_dir = os.path.join(output_dir, "audio")
        self.csv_file = os.path.join(output_dir, "dataset.csv")
        self.sample_rate = sample_rate
        self.duration = duration
        
        # Tạo thư mục nếu chưa có
        os.makedirs(self.audio_dir, exist_ok=True)
        
        self.recording = False
        self.recorded_audio = None
        
    def generate_vietnamese_texts(self, num_texts: int = 20) -> List[str]:
        """
        Sử dụng Gemini API để tạo các câu tiếng Việt đa dạng
        
        Args:
            num_texts: Số lượng câu cần tạo
            
        Returns:
            List các câu tiếng Việt
        """
        prompt = f"""
        Hãy tạo {num_texts} câu tiếng Việt đa dạng để thu âm cho voice conversion dataset.
        
        Yêu cầu:
        - Mỗi câu dài từ 8-15 từ
        - Bao gồm nhiều loại câu: tường thuật, nghi vấn, cảm thán
        - Đa dạng về chủ đề: cuộc sống hàng ngày, công việc, cảm xúc, thời tiết, gia đình
        - Chứa các âm vị khác nhau trong tiếng Việt
        - Có số đếm, tên riêng, từ ghép
        - Tránh từ tục tĩu, nhạy cảm
        - Mỗi câu trên một dòng
        - Không đánh số thứ tự
        
        Ví dụ:
        Hôm nay trời rất đẹp, chúng ta cùng đi dạo công viên nhé
        Bạn có thể giúp tôi giải quyết vấn đề này được không?
        Tôi rất vui khi được gặp lại bạn sau một thời gian dài
        """
        
        try:
            response = self.model.generate_content(prompt)
            texts = [line.strip() for line in response.text.strip().split('\n') 
                    if line.strip() and not line.strip().isdigit()]
            
            # Lọc và làm sạch
            cleaned_texts = []
            for text in texts:
                # Loại bỏ số đầu dòng nếu có
                text = text.lstrip('0123456789.- ')
                if len(text.split()) >= 5 and len(text) <= 150:  # Kiểm tra độ dài hợp lý
                    cleaned_texts.append(text)
            
            return cleaned_texts[:num_texts]
            
        except Exception as e:
            print(f"Lỗi khi tạo text với Gemini API: {e}")
            # Fallback texts nếu API lỗi
            return self._get_fallback_texts(num_texts)
    
    def _get_fallback_texts(self, num_texts: int) -> List[str]:
        """Tạo texts dự phòng nếu API lỗi"""
        fallback_texts = [
            "Xin chào, tôi là trợ lý ảo của bạn hôm nay",
            "Thời tiết hôm nay thật đẹp để đi dạo công viên",
            "Bạn có muốn uống một cốc cà phê không?",
            "Tôi rất thích nghe nhạc vào buổi tối",
            "Gia đình là điều quan trọng nhất trong cuộc sống",
            "Công việc hôm nay có nhiều thử thách thú vị",
            "Chúng ta cùng học tiếng Anh nhé bạn",
            "Món phở Việt Nam rất ngon và nổi tiếng",
            "Tôi đang đọc một cuốn sách hay về lịch sử",
            "Bạn có kế hoạch gì cho ngày mai không?",
            "Trẻ em cần được giáo dục tốt từ sớm",
            "Sức khỏe là tài sản quý giá nhất của con người",
            "Mùa xuân về với hoa đào hoa mai nở rộ",
            "Công nghệ phát triển nhanh chóng trong thời đại này",
            "Tình bạn chân thành là điều đáng trân trọng",
            "Học hỏi là một quá trình suốt đời",
            "Du lịch giúp mở rộng tầm nhìn và kiến thức",
            "Âm nhạc có thể chữa lành tâm hồn",
            "Làm việc nhóm mang lại hiệu quả cao",
            "Bảo vệ môi trường là trách nhiệm của mọi người"
        ]
        return fallback_texts[:num_texts]
    
    def create_dataset_structure(self, speakers: List[str], texts_per_speaker: int = 20) -> pd.DataFrame:
        """
        Tạo cấu trúc dataset CSV
        
        Args:
            speakers: Danh sách tên speakers
            texts_per_speaker: Số câu mỗi speaker
            
        Returns:
            DataFrame chứa thông tin dataset
        """
        dataset_rows = []
        
        for speaker in speakers:
            print(f"Đang tạo texts cho speaker: {speaker}")
            
            # Tạo texts cho speaker này
            texts = self.generate_vietnamese_texts(texts_per_speaker)
            
            # Tạo thư mục cho speaker
            speaker_dir = os.path.join(self.audio_dir, speaker)
            os.makedirs(speaker_dir, exist_ok=True)
            
            # Tạo rows cho CSV
            for i, text in enumerate(texts, 1):
                file_name = f"{speaker}_{i:03d}.wav"
                file_path = os.path.join(speaker_dir, file_name)
                
                dataset_rows.append({
                    'file_path': file_path,
                    'label': text,
                    'speaker': speaker
                })
        
        return pd.DataFrame(dataset_rows)
    
    def save_dataset(self, df: pd.DataFrame) -> None:
        """Lưu dataset vào CSV file"""
        df.to_csv(self.csv_file, index=False, encoding='utf-8')
        print(f"Dataset đã được lưu tại: {self.csv_file}")
        
        # Tạo file thống kê
        stats = {
            'total_samples': len(df),
            'num_speakers': df['speaker'].nunique(),
            'speakers': df['speaker'].unique().tolist(),
            'samples_per_speaker': df.groupby('speaker').size().to_dict(),
            'created_at': datetime.now().isoformat()
        }
        
        stats_file = os.path.join(self.output_dir, "dataset_stats.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"Thống kê dataset: {stats_file}")
    
    def create_recording_script(self, df: pd.DataFrame) -> None:
        """Tạo script để hướng dẫn thu âm"""
        script_file = os.path.join(self.output_dir, "recording_instructions.md")
        
        with open(script_file, 'w', encoding='utf-8') as f:
            f.write("# Hướng dẫn thu âm Voice Dataset\n\n")
            f.write("## Yêu cầu kỹ thuật:\n")
            f.write("- Tần số mẫu: 22050 Hz\n")
            f.write("- Định dạng: WAV, 16-bit\n")
            f.write("- Môi trường: Yên tĩnh, ít tiếng vọng\n")
            f.write("- Khoảng cách mic: 15-20cm từ miệng\n\n")
            
            for speaker in df['speaker'].unique():
                f.write(f"## Speaker: {speaker}\n\n")
                speaker_data = df[df['speaker'] == speaker]
                
                for _, row in speaker_data.iterrows():
                    f.write(f"**File: {os.path.basename(row['file_path'])}**\n")
                    f.write(f"Text: {row['label']}\n\n")
        
        print(f"Script thu âm: {script_file}")
    
    def check_audio_devices(self):
        """Kiểm tra và hiển thị thiết bị âm thanh có sẵn"""
        print("🎧 Thiết bị âm thanh có sẵn:")
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"  {i}: {device['name']} (Input)")
        
        default_device = sd.default.device
        print(f"📍 Thiết bị mặc định: {default_device}")
        return devices
    
    def record_audio(self, duration: int = None) -> np.ndarray:
        """
        Thu âm từ microphone
        
        Args:
            duration: Thời gian thu (giây), None để thu đến khi nhấn Enter
            
        Returns:
            Audio data as numpy array
        """
        if duration is None:
            duration = self.duration
            
        try:
            print("🎤 Bắt đầu thu âm...")
            print("⏱️  Nhấn ENTER để dừng thu âm")
            
            # Thu âm trong background thread
            self.recording = True
            self.recorded_audio = []
            
            def audio_callback(indata, frames, time, status):
                if status:
                    print(f"⚠️  Lỗi audio: {status}")
                if self.recording:
                    self.recorded_audio.append(indata.copy())
            
            # Bắt đầu stream
            with sd.InputStream(samplerate=self.sample_rate, channels=1, 
                              callback=audio_callback, dtype='float32'):
                input("") # Đợi nhấn Enter
            
            self.recording = False
            
            if self.recorded_audio:
                audio_data = np.concatenate(self.recorded_audio, axis=0)
                print("✅ Hoàn thành thu âm")
                return audio_data.flatten()
            else:
                print("❌ Không có dữ liệu âm thanh")
                return np.array([])
                
        except Exception as e:
            print(f"❌ Lỗi khi thu âm: {e}")
            return np.array([])
    
    def play_audio_preview(self, audio_data: np.ndarray):
        """Phát lại âm thanh để kiểm tra"""
        try:
            print("🔊 Đang phát âm thanh...")
            sd.play(audio_data, self.sample_rate)
            sd.wait()
            print("✅ Hoàn thành phát âm thanh")
        except Exception as e:
            print(f"❌ Lỗi khi phát âm thanh: {e}")
    
    def save_audio(self, audio_data: np.ndarray, file_path: str) -> bool:
        """
        Lưu âm thanh vào file
        
        Args:
            audio_data: Dữ liệu âm thanh
            file_path: Đường dẫn file
            
        Returns:
            True nếu thành công
        """
        try:
            # Chuẩn hóa audio về range [-1, 1]
            if len(audio_data) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
                
                # Tạo thư mục nếu cần
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                # Lưu file
                sf.write(file_path, audio_data, self.sample_rate)
                print(f"💾 Đã lưu: {file_path}")
                return True
            else:
                print("❌ Không có dữ liệu để lưu")
                return False
                
        except Exception as e:
            print(f"❌ Lỗi khi lưu audio: {e}")
            return False
    
    def record_and_save_texts(self, speaker: str, texts: List[str]) -> List[Dict]:
        """
        Thu âm và lưu một loạt texts cho một speaker
        
        Args:
            speaker: Tên speaker
            texts: Danh sách texts cần thu âm
            
        Returns:
            List các record đã thu thành công
        """
        print(f"\n🎙️  Bắt đầu thu âm cho speaker: {speaker}")
        print(f"📝 Số câu cần thu: {len(texts)}")
        
        speaker_dir = os.path.join(self.audio_dir, speaker)
        os.makedirs(speaker_dir, exist_ok=True)
        
        recorded_data = []
        
        for i, text in enumerate(texts, 1):
            print(f"\n--- Câu {i}/{len(texts)} ---")
            print(f"📖 Text: {text}")
            
            while True:
                print("\n🎤 Sẵn sàng thu âm...")
                print("Tùy chọn:")
                print("  ENTER - Bắt đầu thu âm")
                print("  s - Skip câu này")
                print("  q - Quit")
                
                choice = input("Chọn: ").strip().lower()
                
                if choice == 'q':
                    print("🛑 Dừng thu âm")
                    return recorded_data
                elif choice == 's':
                    print("⏭️  Skip câu này")
                    break
                elif choice == '' or choice == 'enter':
                    # Thu âm
                    audio_data = self.record_audio()
                    
                    if len(audio_data) > 0:
                        # Phát lại để kiểm tra
                        print("\n🔊 Nghe lại audio vừa thu:")
                        self.play_audio_preview(audio_data)
                        
                        # Xác nhận lưu
                        while True:
                            save_choice = input("\n💾 Lưu audio này? (y/n/r): ").strip().lower()
                            if save_choice == 'y':
                                file_name = f"{speaker}_{i:03d}.wav"
                                file_path = os.path.join(speaker_dir, file_name)
                                
                                if self.save_audio(audio_data, file_path):
                                    recorded_data.append({
                                        'file_path': file_path,
                                        'label': text,
                                        'speaker': speaker
                                    })
                                    print("✅ Đã lưu thành công!")
                                    break
                                else:
                                    print("❌ Lỗi khi lưu, thử lại")
                                    continue
                            elif save_choice == 'n':
                                print("❌ Không lưu, thu lại")
                                break
                            elif save_choice == 'r':
                                print("🔄 Thu lại")
                                break
                            else:
                                print("Nhập y (yes), n (no), hoặc r (record again)")
                        
                        if save_choice == 'y':
                            break
                    else:
                        print("❌ Thu âm thất bại, thử lại")
                else:
                    print("Lựa chọn không hợp lệ")
        
        return recorded_data
    
    def generate_and_record_dataset(self, speakers: List[str], texts_per_speaker: int = 20) -> pd.DataFrame:
        """
        Function chính để tạo và thu âm toàn bộ dataset
        
        Args:
            speakers: Danh sách tên speakers
            texts_per_speaker: Số câu mỗi speaker
            
        Returns:
            DataFrame chứa dataset đã thu âm
        """
        print("🎤 Bắt đầu tạo và thu âm Voice Conversion Dataset...")
        print(f"Speakers: {speakers}")
        print(f"Texts per speaker: {texts_per_speaker}")
        
        # Kiểm tra thiết bị âm thanh
        self.check_audio_devices()
        
        all_recorded_data = []
        
        for speaker in speakers:
            print(f"\n{'='*50}")
            print(f"🎙️  SPEAKER: {speaker}")
            print(f"{'='*50}")
            
            # Tạo texts cho speaker này
            print("📝 Đang tạo texts...")
            texts = self.generate_vietnamese_texts(texts_per_speaker)
            
            if not texts:
                print(f"❌ Không thể tạo texts cho {speaker}")
                continue
            
            # Thu âm các texts
            recorded_data = self.record_and_save_texts(speaker, texts)
            all_recorded_data.extend(recorded_data)
            
            print(f"\n✅ Hoàn thành speaker {speaker}: {len(recorded_data)}/{len(texts)} câu")
        
        # Tạo DataFrame và lưu
        if all_recorded_data:
            df = pd.DataFrame(all_recorded_data)
            self.save_dataset(df)
            
            print(f"\n🎉 HOÀN THÀNH DATASET!")
            print(f"📊 Tổng cộng: {len(df)} samples")
            print(f"🎙️  Speakers: {df['speaker'].nunique()}")
            print(f"📁 Lưu tại: {self.output_dir}")
            
            return df
        else:
            print("❌ Không có dữ liệu nào được thu âm")
            return pd.DataFrame()
    
    def resume_recording(self, csv_file: str = None) -> pd.DataFrame:
        """
        Tiếp tục thu âm từ dataset đã có
        
        Args:
            csv_file: Đường dẫn file CSV hiện có
            
        Returns:
            DataFrame cập nhật
        """
        if csv_file is None:
            csv_file = self.csv_file
            
        if not os.path.exists(csv_file):
            print(f"❌ Không tìm thấy file CSV: {csv_file}")
            return pd.DataFrame()
        
        # Đọc dataset hiện có
        df = pd.read_csv(csv_file)
        print(f"📋 Đã load dataset: {len(df)} records")
        
        # Tìm các file chưa tồn tại
        missing_records = []
        for _, row in df.iterrows():
            if not os.path.exists(row['file_path']):
                missing_records.append(row)
        
        if not missing_records:
            print("✅ Tất cả audio files đã có sẵn")
            return df
        
        print(f"🎤 Cần thu thêm {len(missing_records)} files")
        
        # Group theo speaker
        missing_by_speaker = {}
        for record in missing_records:
            speaker = record['speaker']
            if speaker not in missing_by_speaker:
                missing_by_speaker[speaker] = []
            missing_by_speaker[speaker].append(record)
        
        # Thu âm từng speaker
        for speaker, records in missing_by_speaker.items():
            print(f"\n🎙️  Thu âm thiếu cho speaker: {speaker}")
            texts = [record['label'] for record in records]
            
            recorded_data = self.record_and_save_texts(speaker, texts)
            print(f"✅ Hoàn thành: {len(recorded_data)} files")
        
        # Cập nhật dataset
        self.save_dataset(df)
        return df

# Ví dụ sử dụng
if __name__ == "__main__":
    # Cấu hình
    GEMINI_API_KEY = "AIzaSyAOpTOA26r2zfRpU-5SL430Elu0pY7B3js"  # Thay bằng API key thực
    SPEAKERS = ["thanh", "ducanh", "doan"]  # Danh sách speakers
    TEXTS_PER_SPEAKER = 30
    
    # Tạo generator
    generator = VoiceDatasetGenerator(
        gemini_api_key=GEMINI_API_KEY,
        output_dir="voice_dataset",
        sample_rate=22050,  # Tần số mẫu
        duration=10         # Thời gian thu tối đa
    )
    
    # Kiểm tra thiết bị âm thanh
    generator.check_audio_devices()
    
    # Tùy chọn 1: Tạo mới và thu âm dataset
    print("\n=== TẠO MỚI DATASET ===")
    dataset_df = generator.generate_and_record_dataset(
        speakers=SPEAKERS,
        texts_per_speaker=TEXTS_PER_SPEAKER
    )
    
    # Tùy chọn 2: Tiếp tục thu âm dataset đã có
    # print("\n=== TIẾP TỤC THU ÂM ===")
    # dataset_df = generator.resume_recording()
    
    # Hiển thị preview
    if not dataset_df.empty:
        print("\n📊 Preview dataset:")
        print(dataset_df.head(10))
        print(f"\nShape: {dataset_df.shape}")
        print(f"Speakers: {dataset_df['speaker'].unique()}")
        
        # Thống kê files đã có
        existing_files = sum(1 for _, row in dataset_df.iterrows() 
                           if os.path.exists(row['file_path']))
        print(f"Audio files có sẵn: {existing_files}/{len(dataset_df)}")
    else:
        print("❌ Không có dữ liệu trong dataset")