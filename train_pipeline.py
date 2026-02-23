import json
import logging
import traceback
import subprocess
from pathlib import Path
import ffmpeg
from structure import Task
from config import Config
from pydantic import BaseModel


class TrainPipeline:
    @staticmethod
    def hv_docker_running() -> bool:
        """檢查是否有 Docker 任務執行中"""
        for cmd, image_name in Config.docker_imgs.items():
            if _is_docker_running(image_name):
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
                _process_uvr5_task(task)
            return

        except Exception as e:
            logging.error(f"process 執行期間崩潰: {e}")
            raise e

    @staticmethod
    def chk_standard_task() -> Task:
        """根據時間判斷執行的任務優次"""
        # 檢查時間狀態 (Config 內定義)
        is_night_time: bool = Config.is_night_task_time()

        # 目前優化處理 UVR5
        return _get_uvr5_extract_vocal_task()


def _get_uvr5_extract_vocal_task() -> Task:
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
                    if _is_audio_file(file):
                        # 成功搵到第一個任，封裝成 Task 回傳
                        # character_name 就是 sub_dir 的名字
                        return Task(
                            cmd="UVR5",
                            sub_cmd="extract",
                            file_path=file,
                            character_name=sub_dir.name,
                        )
    return None


def _is_audio_file(file_path: Path) -> bool:
    """使用 ffprobe 檢查是否為有效的音頻檔案"""
    try:
        probe = ffmpeg.probe(str(file_path))
        for stream in probe.get("streams", []):
            if stream.get("codec_type") == "audio":
                return True
    except Exception:
        return False
    return False


def _run_docker(image_name: str, args: list[str]) -> bool:
    # 攞到目前最空閒粒 GPU (例如 "cuda:0")
    device_str, is_half = Config.get_best_device()
    # 轉做 docker 需要嘅 ID (例如 "0")
    device_id = device_str.split(":")[-1] if "cuda" in device_str else "all"

    # 基礎指令
    cmd = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        f"device={device_id}",
        "-e",
        "PYTHONPATH=/app:/app/uvr5",
        "-v",
        f"{Config.dirs['DATA_ROOT']}:{Config.docker_root}",
        image_name,
    ]

    # 增加子參數
    cmd.extend(args)

    try:
        logging.info(f"[{image_name}] 🚀 啟動 Docker 任務...")

        # 唔用 capture_output，改用 stdout=subprocess.PIPE
        # 咁樣可以即時將 Docker 嘅 output 導向到你個 log 檔
        with subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        ) as p:
            for line in p.stdout:
                # 去除換行符號並寫入 logging
                logging.info(f"[{image_name}] [Docker Output] {line.strip()}")

            p.wait()
            if p.returncode == 0:
                logging.info("[{image_name}] ✅ Docker 執行完畢並成功退出")
                return True
            else:
                logging.error(
                    f"[{image_name}] ❌ Docker 報錯退出，Exit Code: {p.returncode}"
                )
                return False

    except Exception as e:
        logging.error(f"[{image_name}] 💥 呼叫 Docker 時發生系統錯誤: {e}")
        return False


def _is_docker_running(image_name: str) -> bool:
    """
    檢查是否有基於該 Image Name 的 Container 正在執行中
    """
    try:
        # 使用 docker ps 過濾 ancestor (祖先鏡像)
        # --format "{{.Image}}" 令輸出只顯示鏡像名
        cmd = [
            "docker",
            "ps",
            "--filter",
            f"ancestor={image_name}",
            "--format",
            "{{.Image}}",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # 如果輸出包含 image_name，代表至少有一個 Container 行緊
        return image_name in result.stdout
    except Exception as e:
        logging.error(f"檢查 Image 狀態時出錯: {e}")
        return False


def _process_uvr5_task(task: Task):
    if not task.in_process:
        task.to_file(Config.train_task_file)
        _run_docker(
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
        
    if task.sub_cmd == "extract":
        # TODO 在 vocal_dir 找 .reformatted_vocals.wav 字尾的 File
        # TODO Check 佢如果是Audio File, 就move to train_dir and rename to "vocal.wav"
        # TODO move file_path to train_dir and rename to "original" + original ext
    
    return
