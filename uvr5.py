import logging
import shutil
from pathlib import Path
from structure import Task
from config import Config
from pydantic import BaseModel

from tools import Tools

class UVR5:
    
    @staticmethod
    def get_uvr5_extract_vocal_task() -> Task:
        """遍歷 input 目錄，尋找需要處理的音頻"""
        base_dir = Config.dirs["TRAIN_INPUT"]

        if not base_dir.exists():
            logging.warning(f"Input 目錄不存在: {base_dir}")
            return None

        # 遍歷次級目錄 (角色名資料夾)
        for sub_dir in sorted(base_dir.iterdir()):
            if sub_dir.is_dir():
                # 遍歷音頻檔案
                for file in sorted(sub_dir.iterdir()):
                    # 排除隱藏檔案同埋非音頻
                    if file.is_file() and not file.name.startswith("."):
                        if Tools.is_audio_file(file):
                            # 成功搵到第一個任，封裝成 Task 回傳
                            # character_name 就是 sub_dir 的名字
                            return Task(
                                cmd="UVR5",
                                sub_cmd="extract",
                                file_path=file,
                                character_name=sub_dir.name,
                            )
        return None 
    
    
    @staticmethod
    def process_uvr5_task(task: Task):
        if not task.in_process:
            task.to_file(Config.train_task_file)
            Tools.run_docker(
                Config.docker_imgs[task.cmd],
                [
                    "--task_type",
                    task.sub_cmd,
                    "--file_path",
                    str(task.docker_file_path),
                    "--vocal_dir",
                    str(task.docker_vocal_dir),
                    "--inst_dir",
                    str(task.docker_inst_dir),
                ],
            )

        file_file_name: str = None
        store_file_name: str = None
        if task.sub_cmd == "extract":
            target_file_name = "*.reformatted_vocals.wav"
            store_file_name = "vocal.wav"

        if not target_file_name is None:
            logging.info(f"正在整理 {task.character_name} 的提取結果...")

            is_find_file: bool = False

            # 1. 在 vocal_dir 搵 target_file_name 字尾嘅 File
            # UVR5 (Roformer) 預設會喺 output folder 產生呢種名嘅 file
            vocal_files = list(task.vocal_dir.glob(target_file_name))
            if vocal_files:
                for vocal_file in vocal_files:
                    # 2. Check 佢係咪有效嘅 Audio File 並搬移
                    if Tools.is_audio_file(vocal_file):
                        dest_vocal = task.train_dir / store_file_name
                        shutil.move(str(vocal_file), str(dest_vocal))
                        logging.info(f"✅ 已提取人聲: {dest_vocal}")
                        is_find_file = True
                        break
                    else:
                        logging.error(f"❌ 搵到嘅人聲檔損毀或格式不正確: {vocal_file}")

            if is_find_file:
                # 3. Check 吓是否 Original Audio File, if yes, move to train directory
                if task.file_path.exists() and task.file_path.parent == task.char_dir:
                    # 4. 搬 file_path (原始音檔) 到 train_dir 並 rename 做 "original" + ext
                    original_ext = task.file_path.suffix  # 例如 .ogg, .mp3, .wav
                    dest_original = task.train_dir / f"original{original_ext}"

                    # 使用 shutil.move 確保跨磁碟搬移都冇問題
                    shutil.move(str(task.file_path), str(dest_original))
                    logging.info(f"📦 原始音檔已備份至: {dest_original}")
                else:
                    logging.error(f"❌ 搵唔到原始音檔，無法搬移: {task.file_path}")
            else:
                logging.warning(f"⚠️ 在 {task.vocal_dir} 找不到 {target_file_name}")
        return
