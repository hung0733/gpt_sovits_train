import os
import torch
import librosa
import ffmpeg
import numpy as np
import soundfile as sf
import traceback
from pathlib import Path
from config import Config
from uvr5.bsroformer import Roformer_Loader
from uvr5.mdxnet import MDXNetDereverb
from uvr5.vr import AudioPre, AudioPreDeEcho

class UVR5Processor:
    """
    一站式 UVR5 處理器。
    注意：vocal_output_path 與 inst_output_path 均為 Folder Path。
    """

    # 用嚟儲存 Loader 實例，避免批次處理時重複 Load 模型到 GPU
    _model_cache = {}

    @staticmethod
    def extract_vocal(
        file_path: Path,
        vocal_output_dir: Path,
        inst_output_dir: Path,
        model_name="model_bs_roformer_ep_317_sdr_12.9755",
    ) -> bool:
        """使用 BS-Roformer 提取人聲"""
        print(f"[提取人聲] 處理中: {file_path.name}")
        rslt = UVR5Processor._process(
            model_name, file_path, vocal_output_dir, inst_output_dir
        )
        return rslt

    @staticmethod
    def dereverb(
        file_path: Path,
        vocal_output_dir: Path,
        inst_output_dir: Path,
        model_name="onnx_dereverb",
    ) -> bool:
        """使用 ONNX 模型去混響"""
        print(f"[去混響] 處理中: {file_path.name}")
        return UVR5Processor._process(
            model_name, file_path, vocal_output_dir, inst_output_dir
        )

    @staticmethod
    def deecho(
        file_path: Path,
        vocal_output_dir: Path,
        inst_output_dir: Path,
        model_name="VR-DeEchoAggressive",
    ) -> bool:
        """使用 PTH 模型去延遲"""
        print(f"[去延遲] 處理中: {file_path.name}")
        return UVR5Processor._process(
            model_name, file_path, vocal_output_dir, inst_output_dir
        )

    # --- Private Methods ---

    @staticmethod
    def _process(
        model_name: str,
        file_path: Path,
        vocal_output_dir: Path,
        inst_output_dir: Path,
    ) -> bool:
        # 既然傳入嚟係 Folder，直接 ensure 呢個 Path 就得
        UVR5Processor._ensure_dir(vocal_output_dir)
        UVR5Processor._ensure_dir(inst_output_dir)

        device, is_half = Config.get_best_device()
        is_hp3 = "HP3" in model_name
        
        # 模型緩存邏輯
        cache_key = f"{model_name}_{device}_{is_half}"
        if cache_key in UVR5Processor._model_cache:
            func = UVR5Processor._model_cache[cache_key]
        else:
            if "onnx_dereverb" in model_name.lower():
                func = MDXNetDereverb(15, str(Config.dirs["UVR5_MODEL"] / (model_name + ".onnx")))
            elif "roformer" in model_name.lower():
                func = Roformer_Loader(
                    str(Config.dirs["UVR5_MODEL"] / (model_name + ".ckpt")),
                    str(Config.dirs["UVR5_MODEL"] / (model_name + ".yaml")),
                    device,
                    is_half,
                )
            elif "DeEcho" not in model_name:
                func = AudioPre(
                    10, Config.dirs["UVR5_MODEL"] / (model_name + ".pth"), device, is_half
                )
            else:
                func = AudioPreDeEcho(
                    10, Config.dirs["UVR5_MODEL"] / (model_name + ".pth"), device, is_half
                )
            UVR5Processor._model_cache[cache_key] = func
        
        if func is not None and file_path.exists():
            done = 0
            try:
                # 預檢音頻格式
                info = ffmpeg.probe(str(file_path), cmd="ffprobe")
                if info["streams"][0]["channels"] == 2 and info["streams"][0].get("sample_rate") == "44100":
                    # 傳入 str 格式嘅 Folder Path
                    func._path_audio_(str(file_path), str(inst_output_dir), str(vocal_output_dir), "wav", is_hp3)
                    done = 1
            except:
                pass
                
            if done == 0:
                # 轉碼處理，確保臨時文件夾存在
                tmp_dir = Path(os.environ.get("TEMP", "/tmp"))
                tmp_path = tmp_dir / f"{file_path.stem}.reformatted.wav"
                
                print(f"轉碼處理中: {file_path.name}")
                os.system(f'ffmpeg -i "{file_path}" -vn -acodec pcm_s16le -ac 2 -ar 44100 "{tmp_path}" -y -loglevel quiet')
                
                try:
                    func._path_audio_(str(tmp_path), str(inst_output_dir), str(vocal_output_dir), "wav", is_hp3)
                    if tmp_path.exists(): tmp_path.unlink()
                except:
                    traceback.print_exc()
                    print(f"處理失敗: {file_path.name}")
                    return False
            return True
        return False

    @staticmethod
    def _ensure_dir(dir_path: Path):
        """確保目錄存在"""
        dir_path.mkdir(parents=True, exist_ok=True)