import json
import logging
import shutil
import traceback
import subprocess
from pathlib import Path
import ffmpeg
from structure import Task
from config import Config
from pydantic import BaseModel
from tools import Tools
from uvr5 import UVR5


class TrainPipeline:
    @staticmethod
    def hv_docker_running() -> bool:
        """檢查是否有 Docker 任務執行中"""
        for cmd, image_name in Config.docker_imgs.items():
            if Tools.is_docker_running(image_name):
                logging.info(f"Docker 任務 [{cmd}] 正在執行中")
                return True
        return False

    @staticmethod
    def chk_process_task() -> Task:
        """檢查是否有處理到一半的任務檔案"""
        file: Path = Config.train_task_file
        if file.exists():
            task: Task = Task.from_file(file)
            task.in_process = True
            return task
        return None

    @staticmethod
    def process(task: Task):
        try:
            if task.cmd == "UVR5":
                UVR5.process_uvr5_task(task)

            if Config.train_task_file.exists():
                Config.train_task_file.unlink()
                logging.info(f"🗑️ 已刪除任務追蹤檔: {Config.train_task_file}")
            return

        except Exception as e:
            logging.error(f"process 執行期間崩潰: {e}")
            raise e

    @staticmethod
    def chk_standard_task() -> Task:
        """根據時間判斷執行的任務優次"""
        # 檢查時間狀態 (Config 內定義)
        is_night_time: bool = Config.is_night_task_time()

        task : Task = None
        
        if task is None:
            task = UVR5.get_uvr5_deecho_vocal_task()
        if task is None:
            task = UVR5.get_uvr5_dereverb_vocal_task()
        if task is None:
            task = UVR5.get_uvr5_extract_vocal_task()
    
        return task










