"""Config and paths for the change_rgb_i20 regression package."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml


def _is_package(p: Path) -> bool:
    return (
        (p / "helpers.py").exists()
        and (p / "config.yaml").exists()
        and (p / "train_lib.py").exists()
        and (p / "cnn.py").exists()
    )


def find_package_root() -> Path:
    here = Path(__file__).resolve().parent
    if _is_package(here):
        return here
    cwd = Path.cwd().resolve()
    candidates = [cwd, cwd / "change_rgb_i20"]
    candidates.extend(cwd.parents)
    for parent in [cwd, *cwd.parents]:
        candidates.append(parent / "change_rgb_i20")
    seen: set[str] = set()
    for p in candidates:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        if _is_package(p):
            return p
    raise FileNotFoundError(
        "找不到 change_rgb_i20（需要 helpers.py、config.yaml、cnn.py、train_lib.py）。"
        f"cwd={cwd}"
    )


def ensure_sys_path(root: Path | None = None) -> Path:
    root = root or find_package_root()
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)
    return root


def _to_namespace(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


def load_config(root: Path | None = None) -> SimpleNamespace:
    root = root or find_package_root()
    with (root / "config.yaml").open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = _to_namespace(raw)
    cfg.project.root = str(root)
    task = str(getattr(getattr(cfg, "cnn", SimpleNamespace()), "task", ""))
    if task != "regression":
        raise RuntimeError(
            f"{root}/config.yaml 的 cnn.task 应为 regression，实际是 {task!r}。"
            "请用 change_rgb_i20 里的 config.yaml。"
        )
    return cfg


def abspath(cfg: SimpleNamespace, rel: str) -> Path:
    p = Path(str(rel)).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (Path(cfg.project.root) / p).resolve()


def make_dirs(cfg: SimpleNamespace) -> dict[str, Path]:
    proc_dir = abspath(cfg, cfg.paths.processed)
    img_dir = abspath(cfg, cfg.paths.images)
    log_dir = abspath(cfg, cfg.paths.logs)
    fig_dir = abspath(cfg, cfg.paths.figures)
    model_dir = abspath(cfg, getattr(cfg.paths, "models", "models"))
    results_dir = abspath(cfg, getattr(cfg.paths, "results", "results"))
    remote = Path(str(getattr(cfg.paths, "images_remote", "") or "")).expanduser()
    output = Path(str(getattr(cfg.paths, "output_remote", "") or "")).expanduser()
    for d in (img_dir, log_dir, fig_dir, model_dir, results_dir):
        d.mkdir(parents=True, exist_ok=True)
    if proc_dir.exists() or not proc_dir.is_absolute():
        proc_dir.mkdir(parents=True, exist_ok=True)
    return {
        "processed": proc_dir,
        "images": img_dir,
        "images_remote": remote,
        "output_remote": output,
        "models": model_dir,
        "results": results_dir,
        "logs": log_dir,
        "figures": fig_dir,
    }


def notebook_bootstrap():
    root = ensure_sys_path(find_package_root())
    cfg = load_config(root)
    paths = make_dirs(cfg)
    return root, cfg, paths
