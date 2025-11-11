from __future__ import annotations
from pathlib import Path
import json

def load_config(repo_root: Path) -> dict:
    cfg_path = repo_root / ".git" / ".code-sim-config.json"
    if not cfg_path.exists():
        return {
            "version": 1,
            "base_dir": ".",
            "include_dirs": [],
            "include_files": [],
            "exclude_patterns": ["**/__pycache__/**", "**/node_modules/**", "**/.git/**"],
            "languages": ["python", "java"],
            "max_files": 200,
        }
    return json.loads(cfg_path.read_text())

def save_config(repo_root: Path, cfg: dict) -> None:
    cfg_path = repo_root / ".git" / ".code-sim-config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))
