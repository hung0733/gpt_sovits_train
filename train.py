import logging
from logging.handlers import TimedRotatingFileHandler
import sys
import traceback
from pathlib import Path
from structure import Task
from config import Config
from tools.tools import Tools
from train_pipeline import TrainPipeline

# 設定 Log 檔案路徑
LOCK_FILE = Path("/tmp/tts_train.lock")

LOG_DIR = Config.dirs["TRAIN_ROOT"] / "log"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "console.log"

Tools.archive_old_logs(LOG_DIR)

# 2. 配置 Logging
file_handler = TimedRotatingFileHandler(
    filename=str(LOG_FILE),
    when="midnight",    # 每日凌晨分割
    interval=1,         # 每 1 日一次
    backupCount=0,      # 手動 archive，唔使自動 delete
    encoding="utf-8"
)

file_handler.suffix = "%Y-%m-%d"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        file_handler,
        logging.StreamHandler(sys.stdout),
    ],
)

def main():
    # 1. 檢查鎖定狀態 (原子操作守門員)
    if LOCK_FILE.exists():
        # 定期巡檢通常用 debug，避免 log 塞滿無謂訊息
        logging.debug("上一個任務仍在執行中，跳過巡檢。")
        return
    
    if TrainPipeline.hv_docker_running():
        return

    # 確保所有 Config 入面定義嘅 path 都存在
    for name, path in Config.dirs.items():
        if not path.exists():
            logging.info(f"正在建立目錄: {path}")
            path.mkdir(parents=True, exist_ok=True)

    task : Task = None

    # 2. 檢查目前是否有執行中的任務
    try:
        task = TrainPipeline.chk_process_task()
    except Exception as e:
        logging.error(f"執行條件檢查出錯: {e}")
        return
    
    if task is None:
        # 3. 尋找待處理任務
        try:
            task = TrainPipeline.chk_standard_task()
        except Exception as e:
            logging.error(f"搜尋任務時發生錯誤: {e}")
            return

    # 4. 如果冇任務，安靜退出
    if task is None:
        return

    # 5. 正式開始處理流程
    try:
        # 獲取鎖定文件
        LOCK_FILE.touch()
        
        logging.info("=" * 60)
        logging.info(f"🚀 啟動任務: [{task.cmd} - {task.sub_cmd}]")
        logging.info(f"   角色: {task.character_name}")
        logging.info(f"   音頻名稱: {task.audio_name}")
        logging.info(f"   檔案: {task.file_path.name}")
        logging.info(f"   執行中的任務: {task.in_process}")
        logging.info("-" * 60)

        # 執行封裝好的流水線邏輯
        # 內部應包含 UVR5 Task Docker 调用、ASR API 调用等
        TrainPipeline.process(task)

        logging.info("✅ 該項流水線任務執行成功。")

    except Exception as e:
        logging.error(f"❌ 流水線執行失敗: {str(e)}")
        # 記錄詳細堆疊追蹤，方便 debug
        logging.error(traceback.format_exc())
    finally:
        # 6. 釋放鎖定
        if LOCK_FILE.exists():
            LOCK_FILE.unlink()
        logging.info("🔚 任務序列結束，已釋放 VRAM 鎖定。")
        logging.info("=" * 60)

if __name__ == "__main__":
    main()