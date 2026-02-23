from datetime import datetime, timedelta
import logging
from pathlib import Path
import shutil
import subprocess
import tarfile

import ffmpeg
from sympy import re
import torch

from config import Config


class Tools:
    @staticmethod
    def run_docker(confs: list[str], image_name: str, args: list[str]) -> bool:
        # 攞到目前最空閒粒 GPU (例如 "cuda:0")
        device_str, is_half = Tools.get_best_device()
        # 轉做 docker 需要嘅 ID (例如 "0")
        device_id = device_str.split(":")[-1] if "cuda" in device_str else "all"

        # 基礎指令
        cmd = [
            "docker",
            "run",
            "--rm",
            "--gpus",
            f"device={device_id}"
        ]
        
        cmd.extend(confs)
        
        cmd.extend([image_name])

        # 增加子參數
        cmd.extend(args)

        try:
            logging.info(f"[{image_name}] 🚀 啟動 Docker 任務...")

            # 唔用 capture_output，改用 stdout=subprocess.PIPE
            # 咁樣可以即時將 Docker 嘅 output 導向到你個 log 檔
            with subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            ) as p:
                for line in p.stdout:
                    # 去除換行符號並寫入 logging
                    logging.info(f"[{image_name}] [Docker Output] {line.strip()}")

                p.wait()
                if p.returncode == 0:
                    logging.info(f"[{image_name}] ✅ Docker 執行完畢並成功退出")
                    return True
                else:
                    logging.error(
                        f"[{image_name}] ❌ Docker 報錯退出，Exit Code: {p.returncode}"
                    )
                    return False

        except Exception as e:
            logging.error(f"[{image_name}] 💥 呼叫 Docker 時發生系統錯誤: {e}")
            return False

    @staticmethod
    def is_docker_running(image_name: str) -> bool:
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
        is_half = best_choice[1] == torch.float16

        print(f"--- 硬件檢測報告 ---")
        print(f"最佳設備: {infer_device}")
        print(f"剩餘顯存: {best_choice[2]/(1024**3):.2f} GB")
        print(f"算力等級: {best_choice[3]}")
        print(f"啟用 FP16: {is_half}")
        print(f"------------------")

        return infer_device, is_half

    @staticmethod
    def is_audio_file(file_path: Path) -> bool:
        """使用 ffprobe 檢查是否為有效的音頻檔案"""
        try:
            probe = ffmpeg.probe(str(file_path))
            for stream in probe.get("streams", []):
                if stream.get("codec_type") == "audio":
                    return True
        except Exception:
            return False
        return False

    @staticmethod
    def clear_folder_contents(folder_path: Path):
        for item in folder_path.iterdir():
            try:
                if item.is_file() or item.is_symlink():
                    item.unlink()  # 刪除檔案或符號連結
                elif item.is_dir():
                    shutil.rmtree(item)  # 遞迴刪除子目錄
            except Exception as e:
                print(f"刪除 {item} 時發生錯誤: {e}")

    @staticmethod
    def archive_old_logs(log_dir):
        """
        搵返上個月嘅舊 Log (格式: console.log.YYYY-MM-*) 並打包
        """
        now = datetime.now()
        # 攞上個月嘅年份同月份 (例如 2026-01)
        first_day_of_this_month = now.replace(day=1)
        last_day_of_last_month = first_day_of_this_month - timedelta(days=1)
        last_month_str = last_day_of_last_month.strftime("%Y-%m")

        archive_name = log_dir / f"logs_{last_month_str}.tar.gz"

        # 如果個壓縮包已經喺度，就唔再重複做
        if archive_name.exists():
            return

        # 搵返所有符合 "console.log.YYYY-MM-*" 格式嘅上個月舊檔
        files_to_archive = [
            f for f in log_dir.glob(f"console.log.{last_month_str}-*") if f.is_file()
        ]

        if files_to_archive:
            print(f"發現上月 Log，正在打包至 {archive_name}...")
            try:
                with tarfile.open(archive_name, "w:gz") as tar:
                    for file in files_to_archive:
                        tar.add(file, arcname=file.name)

                # 確定打包成功後，至刪除舊檔
                for file in files_to_archive:
                    file.unlink()
                print(f"歸檔完成，已清理 {len(files_to_archive)} 個舊檔案。")
            except Exception as e:
                print(f"歸檔過程出錯: {e}")
