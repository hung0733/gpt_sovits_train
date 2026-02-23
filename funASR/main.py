import argparse
import sys
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="FunASR Task Runner (Docker Entrypoint)")
    
    # 1. 路徑參數
    parser.add_argument("--input_dir", type=Path, required=True, help="輸入音頻資料夾 (切片後的 .wav)")
    parser.add_argument("--output_file", type=Path, required=True, help="輸出 .list 檔案的路徑")
    
    # 2. FunASR 運行參數 (完全對接你份 funasr_asr.py 嘅 choices)
    parser.add_argument("--language", type=str, default="yue", choices=["zh", "yue", "auto"], help="識別語言 (粵語用 yue)")
    parser.add_argument("--model_size", type=str, default="large", choices=["large", "small"], help="模型大小")
    parser.add_argument("--precision", type=str, default="float16", choices=["float16", "float32"], help="fp16 或 fp32")
    
    # 掛載後的 ASR 腳本絕對路徑
    asr_script_path = Path("/app/asr/funasr_asr.py")

    args = parser.parse_args()

    # 檢查輸入
    if not args.input_dir.exists():
        print(f"❌ Error: 找不到輸入資料夾 {args.input_dir}")
        sys.exit(1)
    
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"--- FunASR 標註任務啟動 ---")
    print(f"輸入目錄: {args.input_dir}")
    print(f"輸出檔案: {args.output_file}")
    print(f"使用語言: {args.language}")
    print(f"--------------------------")

    # 3. 構建指令 (對接你份 script 嘅 flag: -i, -o, -s, -l, -p)
    cmd = [
        "python3", str(asr_script_path),
        "-i", str(args.input_dir),
        "-o", str(args.output_file),
        "-s", args.model_size,
        "-l", args.language,
        "-p", args.precision
    ]

    try:
        result = subprocess.run(cmd, check=True, text=True)
        if result.returncode == 0:
            print(f"\n✅ SUCCESS: 標註完成。")
            sys.exit(0)
    except subprocess.CalledProcessError as e:
        print(f"🔥 執行出錯: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"💀 崩潰: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()