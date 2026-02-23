from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field, model_validator
from config import Config

class Task(BaseModel):
    # 這些是核心輸入欄位
    cmd: str
    sub_cmd: str
    file_path: Path
    character_name: str
    audio_name: str
    
    # 這些是計算出來的欄位，設為 Optional
    in_process: bool = False
    vocal_dir: Optional[Path] = None
    inst_dir: Optional[Path] = None
    train_dir: Optional[Path] = None
    char_dir: Optional[Path] = None
    
    docker_file_path: Optional[Path] = None
    docker_vocal_dir: Optional[Path] = None
    docker_inst_dir: Optional[Path] = None
    docker_train_dir: Optional[Path] = None

    # Pydantic 專用：初始化後執行路徑計算
    def model_post_init(self, __context):

        # --- Host 路徑邏輯 ---
        self.char_dir = Config.dirs["TRAIN_INPUT"] / self.character_name
        self.train_dir = self.char_dir / self.audio_name
        self.vocal_dir = self.train_dir / "vocal"
        self.inst_dir = self.train_dir / "inst"

        # 自動建立實體資料夾
        self.vocal_dir.mkdir(parents=True, exist_ok=True)
        self.inst_dir.mkdir(parents=True, exist_ok=True)

        # --- Docker 路徑映射 ---
        self.docker_file_path = self._replace_docker_root(self.file_path)
        self.docker_train_dir = self._replace_docker_root(self.train_dir)
        self.docker_vocal_dir = self._replace_docker_root(self.vocal_dir)
        self.docker_inst_dir = self._replace_docker_root(self.inst_dir)

    def _replace_docker_root(self, path: Path) -> Path:
        original_path_str = str(path)
        old_prefix = str(Config.dirs["DATA_ROOT"])
        new_prefix = str(Config.docker_root) # 修正: 統一用 dirs
        return Path(original_path_str.replace(old_prefix, new_prefix))

    # --- JSON Helper ---
    def to_json(self) -> str:
        return self.model_dump_json(indent=4)

    def to_file(self, file_path: Path):
        file_path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def from_file(cls, file_path: Path) -> Optional["Task"]:
        if not file_path.exists():
            return None
        # model_validate_json 會自動觸發 model_post_init 重新計算路徑
        return cls.model_validate_json(file_path.read_text(encoding="utf-8"))