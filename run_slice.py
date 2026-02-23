import logging
import os
from pathlib import Path
import librosa
import numpy as np
import noisereduce as nr
from pydub import AudioSegment
from datetime import timedelta
import io
import soundfile as sf

from config import Config
from structure import Task
from tools.tools import Tools



class Slice:
    @staticmethod
    def get_slick_audio_task() -> Optional[Task]: # 記得 import Optional
        base_dir = Config.dirs["TRAIN_INPUT"]
        if not base_dir.exists():
            logging.warning(f"Input 目錄不存在: {base_dir}")
            return None

        for sub_dir in sorted(base_dir.iterdir()):
            if not sub_dir.is_dir():
                continue

            char_name = sub_dir.name
            
            for audio_dir in sorted(sub_dir.iterdir()):
                if not audio_dir.is_dir():
                    continue

                audio_name = audio_dir.name
                vocal_path = audio_dir / "vocal_main_vocal.wav"
                result_dir: Path = audio_dir / "slice"

                # --- 修正後的 Check 邏輯 ---
                # 1. 來源檔案必須存在
                if not vocal_path.exists():
                    continue
                
                # 2. 判斷是否需要執行 Slice:
                #    如果 slice 資料夾唔存在 OR 入面係空嘅
                is_empty = True
                if result_dir.exists():
                    # any(result_dir.iterdir()) 如果入面有任何 file 會回傳 True
                    if any(result_dir.iterdir()):
                        is_empty = False

                if is_empty:
                    return Task(
                        cmd="Slice_Audio",
                        sub_cmd="",  # 呢度補返就唔會報 Pydantic validation error
                        file_path=vocal_path,
                        character_name=char_name,
                        audio_name=audio_name,
                    )
        return None
                                  
    @staticmethod
    def process_slick_audio_task(task: Task):
        if not task.in_process:   
            task.to_file(Config.train_task_file)
            
        Tools.clear_folder_contents(task.slice_dir)
        Slice.slice_and_denoise(str(task.file_path), str(task.slice_dir))
    
    @staticmethod
    def _format_timestamp(ms):
        """將毫秒轉成 HHMMSSms 格式"""
        td = timedelta(milliseconds=ms)
        total_seconds = int(td.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = int(td.microseconds / 1000)
        return f"{hours:02d}{minutes:02d}{seconds:02d}{milliseconds:03d}"

    @staticmethod
    def slice_and_denoise(input_file, output_dir, min_sec=4, max_sec=10, gap_threshold_sec=1.0, top_db=35):
        os.makedirs(output_dir, exist_ok=True)

        logging.info(f"🚀 啟動降噪 + 拆分流程...")
        
        # 1. 載入音訊
        y, sr = librosa.load(input_file, sr=None)
        
        # 2. 執行降噪
        logging.info("🧹 正在進行 AI 降噪處理...")
        # 加入 stationary=True 通常對 UVR5 剩低嘅底噪效果更好更穩定
        y_denoised = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.8, stationary=True)
        
        # 修正：處理 NaN 數值防止轉換崩潰
        y_denoised = np.nan_to_num(y_denoised)

        # 3. 將降噪後嘅數據轉做 pydub 物件 (用 BytesIO 比較穩陣)
        logging.info("存儲臨時音訊...")
        buffer = io.BytesIO()
        sf.write(buffer, y_denoised, sr, format='WAV')
        buffer.seek(0)
        audio = AudioSegment.from_file(buffer)

        # 4. 搵出「有聲區間」
        # 提高 top_db (e.g., 30) 如果仲係搵唔到聲；降低 (e.g., 40) 如果切得太碎
        intervals = librosa.effects.split(y_denoised, top_db=top_db)
        
        if len(intervals) == 0:
            logging.info(f"❌ 依舊搵唔到人聲。試吓將 top_db 較低啲 (而家係 {top_db})")
            return

        final_segments = []
        curr_start_ms = int(intervals[0][0] / sr * 1000)
        curr_end_ms = int(intervals[0][1] / sr * 1000)

        # 5. 斷句邏輯 (1秒空白必斷)
        for i in range(1, len(intervals)):
            next_start_ms = int(intervals[i][0] / sr * 1000)
            next_end_ms = int(intervals[i][1] / sr * 1000)
            
            gap_duration = next_start_ms - curr_end_ms
            current_total_duration = next_end_ms - curr_start_ms

            if gap_duration >= gap_threshold_sec * 1000 or current_total_duration > max_sec * 1000:
                if curr_end_ms - curr_start_ms >= 1000:
                    final_segments.append((curr_start_ms, curr_end_ms))
                curr_start_ms = next_start_ms
                curr_end_ms = next_end_ms
            else:
                curr_end_ms = next_end_ms

        if curr_end_ms - curr_start_ms >= 1000:
            final_segments.append((curr_start_ms, curr_end_ms))

        # 6. 導出檔案
        logging.info(f"✂️ 準備導出 {len(final_segments)} 段乾淨片段...")
        count = 0
        for start, end in final_segments:
            if (end - start) < 2000: continue
            
            chunk = audio[start:end]
            chunk = chunk.set_frame_rate(44100).set_channels(1).set_sample_width(2)
            
            filename = f"{_format_timestamp(start)}.wav"
            save_path = os.path.join(output_dir, filename)
            chunk.export(save_path, format="wav")
            logging.info(f"  ✨ 已導出: {filename} ({ (end-start)/1000 }s)")
            count += 1

        logging.info(f"\n🎉 任務完成！成功切出 {count} 段。")

if __name__ == "__main__":
    TARGET = "/mnt/data/misc/tts/train/input/F001/1/vocal_main_vocal.wav" 
    OUTPUT = "/tmp/slice_audio"
    
    slice_and_denoise(TARGET, OUTPUT, top_db=30) # 稍微調低 top_db 等佢易啲搵到聲