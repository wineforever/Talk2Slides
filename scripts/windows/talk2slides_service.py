from __future__ import annotations

import argparse
import configparser
import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT_DIR / "backend"
DEFAULT_CONFIG = ROOT_DIR / "talk2slides.ini"
DEFAULT_LOG_PATH = ROOT_DIR / "logs" / "Talk2Slides.service.log"
DEFAULT_SERVICE_NAME = "Talk2Slides"
DEFAULT_DESCRIPTION = "Talk2Slides FastAPI Service"
CLI_OPTIONS: "ServiceCliOptions | None" = None
SERVICE_COMMANDS = {"install", "update", "remove", "start", "stop", "restart", "debug"}


try:
    import servicemanager
    import uvicorn
    import win32event
    import win32service
    import win32serviceutil
except ImportError as exc:
    print(
        f"[ERROR] Missing Windows service dependency: {exc}. "
        "Install pywin32 into the Python environment used for the service.",
        file=sys.stderr,
    )
    raise SystemExit(2)


def cfg_get(
    cfg: configparser.ConfigParser,
    section: str,
    key: str,
    default: Optional[str] = None,
) -> Optional[str]:
    if cfg.has_option(section, key):
        return cfg.get(section, key)
    return default


@dataclass
class ServiceCliOptions:
    service_name: str
    display_name: str
    description: str
    config_path: Path
    project_root: Path
    log_path: Path
    remaining: list[str]


@dataclass
class RuntimeOptions:
    service_name: str
    config_path: Path
    project_root: Path
    backend_dir: Path
    log_path: Path
    host: str
    port: int
    log_level: str
    env: Dict[str, str]


def parse_cli(argv: list[str]) -> ServiceCliOptions:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--service-name", default=DEFAULT_SERVICE_NAME)
    parser.add_argument("--display-name")
    parser.add_argument("--description", default=DEFAULT_DESCRIPTION)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--project-root", default=str(ROOT_DIR))
    parser.add_argument("--log-path", default=str(DEFAULT_LOG_PATH))
    known, remaining = parser.parse_known_args(argv)

    service_name = known.service_name.strip() or DEFAULT_SERVICE_NAME
    display_name = (known.display_name or service_name).strip() or service_name

    return ServiceCliOptions(
        service_name=service_name,
        display_name=display_name,
        description=known.description,
        config_path=Path(known.config).expanduser().resolve(),
        project_root=Path(known.project_root).expanduser().resolve(),
        log_path=Path(known.log_path).expanduser().resolve(),
        remaining=remaining,
    )


def get_service_option(service_name: str, option_name: str, fallback: Optional[str]) -> Optional[str]:
    try:
        value = win32serviceutil.GetServiceCustomOption(service_name, option_name, fallback)
    except Exception:
        value = fallback
    if value is None:
        return None
    return str(value)


def load_runtime_options(service_name: str) -> RuntimeOptions:
    if CLI_OPTIONS is None:
        raise RuntimeError("CLI options were not initialized.")

    config_text = get_service_option(service_name, "ConfigPath", str(CLI_OPTIONS.config_path))
    project_root_text = get_service_option(service_name, "ProjectRoot", str(CLI_OPTIONS.project_root))
    log_path_text = get_service_option(service_name, "LogPath", str(CLI_OPTIONS.log_path))

    config_path = Path(config_text).expanduser().resolve()
    project_root = Path(project_root_text).expanduser().resolve()
    backend_dir = project_root / "backend"
    log_path = Path(log_path_text).expanduser().resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not backend_dir.exists():
        raise FileNotFoundError(f"Backend directory not found: {backend_dir}")

    cfg = configparser.ConfigParser()
    cfg.read(config_path, encoding="utf-8")

    host = cfg_get(cfg, "server", "host", "0.0.0.0") or "0.0.0.0"
    port = int(cfg_get(cfg, "server", "port", "8000") or "8000")
    log_level = (cfg_get(cfg, "server", "log_level", "info") or "info").lower()

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    if cfg.has_section("env"):
        for key, value in cfg.items("env"):
            env[key.upper()] = value

    return RuntimeOptions(
        service_name=service_name,
        config_path=config_path,
        project_root=project_root,
        backend_dir=backend_dir,
        log_path=log_path,
        host=host,
        port=port,
        log_level=log_level,
        env=env,
    )


def write_service_options(options: ServiceCliOptions) -> None:
    options.log_path.parent.mkdir(parents=True, exist_ok=True)
    win32serviceutil.SetServiceCustomOption(options.service_name, "ConfigPath", str(options.config_path))
    win32serviceutil.SetServiceCustomOption(options.service_name, "ProjectRoot", str(options.project_root))
    win32serviceutil.SetServiceCustomOption(options.service_name, "LogPath", str(options.log_path))
    print(f"[INFO] Saved service parameters for {options.service_name}")


def build_service_exe_args(options: ServiceCliOptions) -> str:
    script_path = Path(__file__).resolve()
    return f'"{script_path}" --service-name "{options.service_name}"'


class Talk2SlidesService(win32serviceutil.ServiceFramework):
    _svc_name_ = DEFAULT_SERVICE_NAME
    _svc_display_name_ = DEFAULT_SERVICE_NAME
    _svc_description_ = DEFAULT_DESCRIPTION

    def __init__(self, args: list[str]) -> None:
        super().__init__(args)
        self.stop_event = win32event.CreateEvent(None, 0, 0, None)
        self.server: Optional[uvicorn.Server] = None
        self.server_thread: Optional[threading.Thread] = None
        self.server_exception: Optional[BaseException] = None
        self.server_traceback: str = ""
        self.log_stream = None

    def SvcStop(self) -> None:
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        if self.server is not None:
            self.server.should_exit = True
        win32event.SetEvent(self.stop_event)

    def SvcDoRun(self) -> None:
        default_log = str(CLI_OPTIONS.log_path) if CLI_OPTIONS is not None else str(DEFAULT_LOG_PATH)
        early_log_path = Path(get_service_option(self._svc_name_, "LogPath", default_log)).expanduser().resolve()
        self._configure_logging(early_log_path)

        try:
            runtime = load_runtime_options(self._svc_name_)
            self._log(f"Service starting with config: {runtime.config_path}")
            self._run_server(runtime)
        except Exception:
            self._log(traceback.format_exc())
            servicemanager.LogErrorMsg(
                f"{self._svc_name_} failed to start.\n{traceback.format_exc()}"
            )
            raise
        finally:
            self._close_logging()

    def _run_server(self, runtime: RuntimeOptions) -> None:
        self.ReportServiceStatus(win32service.SERVICE_START_PENDING)

        for key, value in runtime.env.items():
            os.environ[key] = value

        os.chdir(runtime.backend_dir)
        backend_dir_text = str(runtime.backend_dir)
        if backend_dir_text not in sys.path:
            sys.path.insert(0, backend_dir_text)

        config = uvicorn.Config(
            "app.main:app",
            host=runtime.host,
            port=runtime.port,
            log_level=runtime.log_level,
            reload=False,
        )
        self.server = uvicorn.Server(config)
        self.server.install_signal_handlers = lambda: None
        self.server_thread = threading.Thread(target=self._server_worker, name="talk2slides-uvicorn", daemon=True)
        self.server_thread.start()

        deadline = time.time() + 10
        while time.time() < deadline and self.server_thread.is_alive() and self.server_exception is None:
            time.sleep(0.25)

        if self.server_exception is not None:
            raise RuntimeError(f"Uvicorn failed during startup:\n{self.server_traceback}") from self.server_exception
        if self.server_thread is None or not self.server_thread.is_alive():
            raise RuntimeError("Uvicorn exited during startup. Check the service log for details.")

        self.ReportServiceStatus(win32service.SERVICE_RUNNING)
        self._log(f"Service listening on http://{runtime.host}:{runtime.port}")
        servicemanager.LogInfoMsg(f"{self._svc_name_} is running.")

        while True:
            wait_result = win32event.WaitForSingleObject(self.stop_event, 1000)
            if wait_result == win32event.WAIT_OBJECT_0:
                break
            if self.server_thread is not None and not self.server_thread.is_alive():
                if self.server_exception is not None:
                    raise RuntimeError(f"Uvicorn exited unexpectedly:\n{self.server_traceback}") from self.server_exception
                raise RuntimeError("Uvicorn exited unexpectedly without an exception.")

        if self.server is not None:
            self.server.should_exit = True

        if self.server_thread is not None:
            self.server_thread.join(timeout=30)

        self._log("Service stopped.")
        servicemanager.LogInfoMsg(f"{self._svc_name_} stopped.")

    def _server_worker(self) -> None:
        try:
            if self.server is None:
                raise RuntimeError("Uvicorn server was not initialized.")
            self.server.run()
        except BaseException as exc:
            self.server_exception = exc
            self.server_traceback = traceback.format_exc()
            self._log(self.server_traceback)

    def _configure_logging(self, log_path: Path) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_stream = log_path.open("a", encoding="utf-8", buffering=1)
        sys.stdout = self.log_stream
        sys.stderr = self.log_stream

    def _close_logging(self) -> None:
        if self.log_stream is None:
            return
        try:
            self.log_stream.flush()
        finally:
            self.log_stream.close()
            self.log_stream = None

    def _log(self, message: str) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}", flush=True)


def main() -> int:
    global CLI_OPTIONS

    CLI_OPTIONS = parse_cli(sys.argv[1:])
    Talk2SlidesService._svc_name_ = CLI_OPTIONS.service_name
    Talk2SlidesService._svc_display_name_ = CLI_OPTIONS.display_name
    Talk2SlidesService._svc_description_ = CLI_OPTIONS.description
    Talk2SlidesService._exe_name_ = sys.executable
    Talk2SlidesService._exe_args_ = build_service_exe_args(CLI_OPTIONS)

    if CLI_OPTIONS.remaining and CLI_OPTIONS.remaining[0].lower() == "configure":
        write_service_options(CLI_OPTIONS)
        return 0

    if CLI_OPTIONS.remaining and CLI_OPTIONS.remaining[0].lower() in SERVICE_COMMANDS:
        sys.argv = [sys.argv[0], *CLI_OPTIONS.remaining]
        return int(win32serviceutil.HandleCommandLine(Talk2SlidesService))

    servicemanager.Initialize()
    servicemanager.PrepareToHostSingle(Talk2SlidesService)
    servicemanager.StartServiceCtrlDispatcher()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
