"""YAML configuration loading and project-relative path helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.yaml"


def load_config(config_path: str | Path = DEFAULT_CONFIG) -> tuple[dict[str, Any], Path]:
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"配置文件根节点必须是mapping: {path}")
    for section in ("model", "preprocess", "inference", "test", "export"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"缺少配置段或格式错误: {section}")
    return config, path


def required(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ValueError(f"缺少配置项: {context}.{key}")
    return mapping[key]


def resolve_config_path(value: str | Path, config_path: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()


def as_float_triplet(value: Any, name: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name}必须包含3个数值")
    return tuple(float(item) for item in value)  # type: ignore[return-value]


def as_int_triplet(value: Any, name: str) -> tuple[int, int, int]:
    result = as_float_triplet(value, name)
    integers = tuple(int(item) for item in result)
    if any(item < 0 or item > 255 for item in integers):
        raise ValueError(f"{name}必须位于0到255")
    return integers  # type: ignore[return-value]
