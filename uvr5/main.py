import argparse
import sys
from pathlib import Path

# 注意：需要確保 UVR5Processor 同 Config 喺 Python Path 入面
from uvr5_processor import UVR5Processor
from config import Config

def main():
    parser = argparse.ArgumentParser(description="UVR5 Task Runner (Docker Mode)")
    # 將 type 設為 Path，argparse 會自動幫你做轉換
    parser.add_argument("--file_path", type=Path, required=True, help="輸入音頻檔案的完整路徑")
    parser.add_argument("--vocal_dir", type=Path, required=True, help="人聲輸出資料夾路徑") 
    parser.add_argument("--inst_dir", type=Path, required=True, help="伴奏輸出資料夾路徑") 
    parser.add_argument("--task_type", type=str, choices=["extract", "dereverb", "deecho"], required=True)
    parser.add_argument("--model_name", type=str, default=None)
    
    args = parser.parse_args()

    # 檢查輸入檔案是否存在
    if not args.file_path.exists():
        print(f"Error: 找不到檔案 {args.file_path}")
        sys.exit(1)

    print(f"--- UVR5 任務啟動 ---")
    print(f"任務類型: {args.task_type}")
    print(f"輸入檔案: {args.file_path.name}")
    print(f"輸出人聲: {args.vocal_dir}")
    print(f"輸出伴奏: {args.inst_dir}")
    print(f"--------------------")

    success = False
    try:
        if args.task_type == "extract":
            # 傳入 Path 物件
            success = UVR5Processor.extract_vocal(
                args.file_path, 
                args.vocal_dir, 
                args.inst_dir, 
                args.model_name or "model_bs_roformer_ep_317_sdr_12.9755"
            )
        elif args.task_type == "dereverb":
            success = UVR5Processor.dereverb(
                args.file_path, 
                args.vocal_dir, 
                args.inst_dir, 
                args.model_name or "onnx_dereverb"
            )
        elif args.task_type == "deecho":
            success = UVR5Processor.deecho(
                args.file_path, 
                args.vocal_dir, 
                args.inst_dir, 
                args.model_name or "VR-DeEchoAggressive"
            )
    except Exception as e:
        print(f"執行期間發生崩潰: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    if success:
        print(f"SUCCESS: {args.file_path.name} 處理完成。")
        print("Container 準備結束並釋放 VRAM...")
        sys.exit(0)
    else:
        print(f"FAILED: {args.file_path.name} 處理失敗。")
        sys.exit(1)

if __name__ == "__main__":
    main()