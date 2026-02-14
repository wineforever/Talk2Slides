#!/usr/bin/env python3
"""Windows launcher for Talk2Slides.

Reads root INI config and supports CLI overrides.
"""

from __future__ import annotations

import argparse
import configparser
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT_DIR / "talk2slides.ini"
BACKEND_DIR = ROOT_DIR / "backend"


def parse_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def cfg_get(cfg: configparser.ConfigParser, section: str, key: str, default: Optional[str] = None) -> Optional[str]:
    if cfg.has_option(section, key):
        return cfg.get(section, key)
    return default


def pick_python(launcher_python: str, use_venv: bool, venv_dir: str) -> str:
    if use_venv:
        venv_python = (ROOT_DIR / venv_dir / "Scripts" / "python.exe").resolve()
        if venv_python.exists():
            return str(venv_python)
        print(f"[WARN] venv python not found: {venv_python}")
    return launcher_python


def parse_set_pairs(items: Iterable[str]) -> Dict[str, str]:
    pairs: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--set expects KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"--set key cannot be empty: {item}")
        pairs[key] = value
    return pairs


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Start Talk2Slides on Windows.")
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to INI config file.")

    # launcher/server
    p.add_argument("--python", help="Python executable path.")
    p.add_argument("--use-venv", help="true/false")
    p.add_argument("--venv-dir", help="Virtual environment directory, relative to repo root.")
    p.add_argument("--auto-install-requirements", help="true/false")
    p.add_argument("--host")
    p.add_argument("--port", type=int)
    p.add_argument("--reload", help="true/false")
    p.add_argument("--log-level")

    # common env overrides
    p.add_argument("--debug", help="true/false")
    p.add_argument("--ffmpeg-path")
    p.add_argument("--libreoffice-path")
    p.add_argument("--model", help="Sentence-Transformer model name.")
    p.add_argument("--similarity-threshold", type=float)
    p.add_argument("--min-display-duration", type=float)
    p.add_argument("--output-resolution")
    p.add_argument("--ppt-export-dpi", type=int)
    p.add_argument("--srt-merge-gap-sec", type=float)
    p.add_argument("--srt-min-duration-sec", type=float)
    p.add_argument("--align-max-backtrack", type=int)
    p.add_argument("--align-max-forward-jump", type=int)
    p.add_argument("--align-switch-penalty", type=float)
    p.add_argument("--align-backtrack-penalty", type=float)
    p.add_argument("--align-forward-jump-penalty", type=float)
    p.add_argument("--align-enforce-sequential", help="true/false")
    p.add_argument("--align-require-full-coverage", help="true/false")
    p.add_argument("--align-keep-short-segments-for-coverage", help="true/false")
    p.add_argument("--video-force-first-slide-frame", help="true/false")

    # arbitrary env overrides
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra environment override, can be repeated.",
    )
    p.add_argument("--dry-run", action="store_true", help="Print launch command and exit.")
    return p


def main() -> int:
    args = build_parser().parse_args()
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"[ERROR] Config not found: {config_path}")
        return 1

    cfg = configparser.ConfigParser()
    cfg.read(config_path, encoding="utf-8")

    launcher_python = args.python or cfg_get(cfg, "launcher", "python", "python")
    use_venv = parse_bool(args.use_venv, parse_bool(cfg_get(cfg, "launcher", "use_venv", "true"), True))
    venv_dir = args.venv_dir or cfg_get(cfg, "launcher", "venv_dir", r"backend\venv")
    auto_install = parse_bool(
        args.auto_install_requirements,
        parse_bool(cfg_get(cfg, "launcher", "auto_install_requirements", "false"), False),
    )

    host = args.host or cfg_get(cfg, "server", "host", "0.0.0.0")
    port = args.port if args.port is not None else int(cfg_get(cfg, "server", "port", "8000"))
    reload_enabled = parse_bool(args.reload, parse_bool(cfg_get(cfg, "server", "reload", "true"), True))
    log_level = (args.log_level or cfg_get(cfg, "server", "log_level", "info")).lower()

    python_exec = pick_python(launcher_python, use_venv, venv_dir)

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"

    # Populate env from [env] section first.
    if cfg.has_section("env"):
        for key, value in cfg.items("env"):
            env[key.upper()] = value

    # CLI overrides for env-mapped fields.
    cli_env_updates = {
        "DEBUG": args.debug,
        "FFMPEG_PATH": args.ffmpeg_path,
        "LIBREOFFICE_PATH": args.libreoffice_path,
        "SENTENCE_TRANSFORMER_MODEL": args.model,
        "DEFAULT_SIMILARITY_THRESHOLD": args.similarity_threshold,
        "DEFAULT_MIN_DISPLAY_DURATION": args.min_display_duration,
        "DEFAULT_OUTPUT_RESOLUTION": args.output_resolution,
        "PPT_EXPORT_DPI": args.ppt_export_dpi,
        "SRT_MERGE_GAP_SEC": args.srt_merge_gap_sec,
        "SRT_MIN_DURATION_SEC": args.srt_min_duration_sec,
        "ALIGN_MAX_BACKTRACK": args.align_max_backtrack,
        "ALIGN_MAX_FORWARD_JUMP": args.align_max_forward_jump,
        "ALIGN_SWITCH_PENALTY": args.align_switch_penalty,
        "ALIGN_BACKTRACK_PENALTY": args.align_backtrack_penalty,
        "ALIGN_FORWARD_JUMP_PENALTY": args.align_forward_jump_penalty,
        "ALIGN_ENFORCE_SEQUENTIAL": args.align_enforce_sequential,
        "ALIGN_REQUIRE_FULL_COVERAGE": args.align_require_full_coverage,
        "ALIGN_KEEP_SHORT_SEGMENTS_FOR_COVERAGE": args.align_keep_short_segments_for_coverage,
        "VIDEO_FORCE_FIRST_SLIDE_FRAME": args.video_force_first_slide_frame,
    }
    for key, value in cli_env_updates.items():
        if value is not None:
            env[key] = str(value)

    # Generic overrides via --set KEY=VALUE
    try:
        generic_updates = parse_set_pairs(args.set)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return 2
    env.update(generic_updates)

    cmd = [
        python_exec,
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        host,
        "--port",
        str(port),
        "--log-level",
        log_level,
    ]
    if reload_enabled:
        cmd.append("--reload")

    if args.dry_run:
        print("[INFO] Dry run mode")
        print(f"[INFO] Python: {python_exec}")
        print(f"[INFO] Backend dir: {BACKEND_DIR}")
        print(f"[INFO] Command: {' '.join(cmd)}")
        return 0

    if auto_install:
        req = BACKEND_DIR / "requirements.txt"
        if req.exists():
            print(f"[INFO] Installing requirements from {req}")
            rc = subprocess.call([python_exec, "-m", "pip", "install", "-r", str(req)], cwd=str(BACKEND_DIR), env=env)
            if rc != 0:
                return rc
        else:
            print(f"[WARN] requirements.txt not found: {req}")

    print("[INFO] Launching Talk2Slides")
    print(f"[INFO] Python: {python_exec}")
    print(f"[INFO] Backend dir: {BACKEND_DIR}")
    print(f"[INFO] Server: http://{host}:{port} (reload={reload_enabled})")

    return subprocess.call(cmd, cwd=str(BACKEND_DIR), env=env)


if __name__ == "__main__":
    raise SystemExit(main())
