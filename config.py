import torch
from datetime import datetime
from pathlib import Path

class Config:
    """
    一站式訓練全局配置類
    """
    dirs = {}
    
    docker_imgs = {
        "UVR5": "uvr5"
    }
    
    # 基礎路徑設定
    _data_root = Path("/mnt/data/misc/tts")
    docker_root = Path("/data")
    
    dirs["DATA_ROOT"] = _data_root
    dirs["TRAIN_ROOT"] = _data_root / "train"
    dirs["TRAIN_INPUT"] = dirs["TRAIN_ROOT"] / "input"
    train_task_file = dirs["TRAIN_ROOT"] / ".task.json"
    

    @staticmethod
    def is_night_task_time() -> bool:
        """
        檢查現時是否處於凌晨 2:00 至 早上 6:00 之間。
        (適合用嚟跑啲消耗極大資源嘅訓練任務)
        """
        current_hour = datetime.now().hour
        return 2 <= current_hour < 6