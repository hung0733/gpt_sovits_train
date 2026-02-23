import torch
from datetime import datetime
from pathlib import Path

class Config:
    """
    一站式訓練全局配置類
    """
    dirs = {}
    
    # 基礎路徑設定
    data_root = Path("/data")
    app_root = Path("/app")

    dirs["DATA_ROOT"] = data_root
    dirs["TRAIN_ROOT"] = data_root / "train"
    dirs["TRAIN_INPUT"] = dirs["TRAIN_ROOT"] / "input"
    dirs["TRAIN_OUTPUT"] = dirs["TRAIN_ROOT"] / "output"
    dirs["UVR5_MODEL"] = app_root / "uvr5" / "uvr5_weights"

    @staticmethod
    def is_night_task_time() -> bool:
        """
        檢查現時是否處於凌晨 2:00 至 早上 6:00 之間。
        (適合用嚟跑啲消耗極大資源嘅訓練任務)
        """
        current_hour = datetime.now().hour
        return 2 <= current_hour < 6
    
    @staticmethod
    def get_best_device():
        """
        自動搵出目前剩餘顯存 (Free VRAM) 最多、且算力最強嘅 GPU。
        回傳: (str: device_name, bool: is_half)
        """
        if not torch.cuda.is_available():
            print("CUDA 不可用，將使用 CPU 推理。")
            return "cpu", False

        tmp = []
        for i in range(torch.cuda.device_count()):
            try:
                # 攞到 (剩餘顯存, 總顯存) 單位係 bytes
                free_mem, total_mem = torch.cuda.mem_get_info(i)
                
                # 攞算力等級 (Compute Capability)
                prop = torch.cuda.get_device_properties(i)
                capability = prop.major + prop.minor / 10
                
                # 算力 >= 7.0 (Volta 架構或之後，如 V100, RTX 20/30/40 系列) 支援 FP16 加速
                supported_dtype = torch.float16 if capability >= 7.0 else torch.float32
                
                # 儲存格式: (device_id, dtype, free_mem, capability)
                tmp.append((f"cuda:{i}", supported_dtype, free_mem, capability))
            except Exception as e:
                print(f"查詢 GPU:{i} 資訊失敗: {e}")

        if not tmp:
            return "cpu", False

        # 排序邏輯：優先比剩餘顯存 (x[2])，其次比算力等級 (x[3])
        best_choice = max(tmp, key=lambda x: (x[2], x[3]))
        
        infer_device = best_choice[0]
        is_half = (best_choice[1] == torch.float16)
        
        print(f"--- 硬件檢測報告 ---")
        print(f"最佳設備: {infer_device}")
        print(f"剩餘顯存: {best_choice[2]/(1024**3):.2f} GB")
        print(f"算力等級: {best_choice[3]}")
        print(f"啟用 FP16: {is_half}")
        print(f"------------------")
        
        return infer_device, is_half

# 預先執行一次初始化測試（可選）
if __name__ == "__main__":
    Config.get_best_device()