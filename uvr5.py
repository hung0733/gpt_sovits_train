import logging
import shutil
from pathlib import Path
from structure import Task
from config import Config
from pydantic import BaseModel

from tools import Tools


class UVR5:

    @staticmethod
    def _find_task_in_folders(sub_cmd: str, input_filename: str, output_filename: str =  None) -> Task:
        """
        通用遍歷邏輯
        :param sub_cmd: 子命令 (extract/dereverb/deecho)
        :param input_filename: 來源檔名 (extract 模式下不適用)
        :param output_filename: 產出檔名 (用嚟 check 係咪處理咗)
        """
        base_dir = Config.dirs["TRAIN_INPUT"]
        if not base_dir.exists():
            logging.warning(f"Input 目錄不存在: {base_dir}")
            return None

        for sub_dir in sorted(base_dir.iterdir()):
            if not sub_dir.is_dir():
                continue
            
            char_name = sub_dir.name

            # 處理 extract：直接喺角色目錄搵 file
            if sub_cmd == "extract":
                for file in sorted(sub_dir.iterdir()):
                    if file.is_file() and not file.name.startswith(".") and Tools.is_audio_file(file):
                        return Task(
                            cmd="UVR5",
                            sub_cmd="extract",
                            file_path=file,
                            character_name=char_name,
                            audio_name=file.stem
                        )
            
            # 處理 deecho / dereverb：喺 audio_dir 入面搵
            else:
                for audio_dir in sorted(sub_dir.iterdir()):
                    if not audio_dir.is_dir():
                        continue
                    
                    audio_name = audio_dir.name
                    vocal_path = audio_dir / input_filename
                    result_path = audio_dir / output_filename if output_filename else None

                    # Check 來源存在 同埋 結果未存在
                    if vocal_path.exists() and Tools.is_audio_file(vocal_path):
                        if result_path and result_path.exists() and Tools.is_audio_file(result_path):
                            continue # 做咗喇，跳過
                        
                        return Task(
                            cmd="UVR5",
                            sub_cmd=sub_cmd,
                            file_path=vocal_path,
                            character_name=char_name,
                            audio_name=audio_name,
                        )
        return None

    @staticmethod
    def get_uvr5_deecho_vocal_task() -> Task:
        return UVR5._find_task_in_folders("deecho", "main_vocal.wav", "vocal_main_vocal.wav")

    @staticmethod
    def get_uvr5_dereverb_vocal_task() -> Task:
        return UVR5._find_task_in_folders("dereverb", "vocal.wav", "main_vocal.wav")

    @staticmethod
    def get_uvr5_extract_vocal_task() -> Task:
        return UVR5._find_task_in_folders("extract", "")

    @staticmethod
    def process_uvr5_task(task: Task):
        if not task.in_process:
            Tools.clear_folder_contents(task.vocal_dir)
            Tools.clear_folder_contents(task.inst_dir)

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

        find_file_name: str = None
        store_file_name: str = None
        if task.sub_cmd == "extract":
            find_file_name = "*.reformatted_vocals.wav"
            store_file_name = "vocal.wav"
        elif task.sub_cmd == "dereverb":
            find_file_name = "*.wav_main_vocal.wav"
            store_file_name = "main_vocal.wav"
        elif task.sub_cmd == "deecho":
            find_file_name = "*.wav_10.wav"
            store_file_name = "vocal_main_vocal.wav"

        if not find_file_name is None:
            logging.info(f"正在整理 {task.character_name} 的提取結果...")

            is_find_file: bool = False

            # 1. 在 vocal_dir 搵 target_file_name 字尾嘅 File
            # UVR5 (Roformer) 預設會喺 output folder 產生呢種名嘅 file
            vocal_files = list(task.vocal_dir.glob(find_file_name))
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
                # 3. Check 吓是否有 Original Audio File, if yes, move to train directory
                if task.file_path.exists() and task.file_path.parent == task.char_dir:
                    # 4. 搬 file_path (原始音檔) 到 train_dir 並 rename 做 "original" + ext
                    original_ext = task.file_path.suffix  # 例如 .ogg, .mp3, .wav
                    dest_original = task.train_dir / f"original{original_ext}"

                    # 使用 shutil.move 確保跨磁碟搬移都冇問題
                    shutil.move(str(task.file_path), str(dest_original))
                    logging.info(f"📦 原始音檔已備份至: {dest_original}")
            else:
                logging.warning(f"⚠️ 在 {task.vocal_dir} 找不到 {find_file_name}")
        return
