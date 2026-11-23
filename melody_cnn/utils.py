from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_path(root: Path, p: str | Path) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = (root / path).resolve()
    return path
