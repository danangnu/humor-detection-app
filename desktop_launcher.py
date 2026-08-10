from __future__ import annotations

import ctypes
import os
import shutil
import socket
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path


APP_NAME = "Humor Bot"
DEFAULT_PORT = 8000


def exe_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def resource_root() -> Path:
    return Path(
        getattr(
            sys,
            "_MEIPASS",
            Path(__file__).resolve().parent,
        )
    ).resolve()


def runtime_root() -> Path:
    base = Path(
        os.environ.get(
            "LOCALAPPDATA",
            str(
                Path.home()
                / "AppData"
                / "Local"
            ),
        )
    )

    return (
        base
        / "HumorBot"
    ).resolve()


def log_path() -> Path:
    path = (
        runtime_root()
        / "logs"
        / "HumorBot.log"
    )
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    return path


def write_log(message: str) -> None:
    stamp = time.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    try:
        with log_path().open(
            "a",
            encoding="utf-8",
        ) as handle:
            handle.write(
                f"[{stamp}] {message}\n"
            )
    except Exception:
        pass


def show_error(message: str) -> None:
    write_log(
        "ERROR: " + message
    )

    if os.name == "nt":
        try:
            ctypes.windll.user32.MessageBoxW(
                0,
                message,
                APP_NAME,
                0x10,
            )
            return
        except Exception:
            pass


def prepare_windowed_stdio() -> None:
    try:
        if sys.stdout is None:
            sys.stdout = open(
                os.devnull,
                "w",
                encoding="utf-8",
            )

        if sys.stderr is None:
            sys.stderr = open(
                os.devnull,
                "w",
                encoding="utf-8",
            )
    except Exception:
        pass


def copy_if_missing(
    source: Path,
    destination: Path,
) -> None:
    if destination.exists():
        return

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if source.is_dir():
        shutil.copytree(
            source,
            destination,
        )
    else:
        shutil.copy2(
            source,
            destination,
        )


def bootstrap_runtime() -> None:
    """
    Keep all mutable ML/admin state outside PyInstaller's 'i' directory.

    First startup copies the packaged baseline model and protected datasets to:
        %LOCALAPPDATA%\\HumorBot\\

    Later promotions, rollbacks and retraining operate only there.
    """
    target = runtime_root()
    target.mkdir(
        parents=True,
        exist_ok=True,
    )

    bootstrap = (
        resource_root()
        / "bootstrap"
    )

    required = [
        (
            bootstrap
            / "models"
            / "humor_transformer",
            target
            / "models"
            / "humor_transformer",
        ),
        (
            bootstrap
            / "data"
            / "processed",
            target
            / "data"
            / "processed",
        ),
    ]

    for source, destination in required:
        if not source.exists():
            raise RuntimeError(
                f"Packaged bootstrap resource is missing: {source}"
            )

        copy_if_missing(
            source,
            destination,
        )

    optional_ood = (
        bootstrap
        / "data"
        / "ood"
    )

    if optional_ood.exists():
        copy_if_missing(
            optional_ood,
            target
            / "data"
            / "ood",
        )

    (
        target
        / "models"
        / "candidates"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        target
        / "models"
        / "model_history"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        target
        / "data"
        / "admin"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        target
        / "data"
        / "retraining"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )


def load_external_env() -> None:
    try:
        from dotenv import load_dotenv

        # Prefer a client-visible .env beside the EXE.
        load_dotenv(
            exe_dir() / ".env",
            override=False,
        )

        # Also support persistent local configuration.
        load_dotenv(
            runtime_root()
            / ".env",
            override=False,
        )
    except Exception as exc:
        write_log(
            f"Could not load .env: {exc}"
        )


def choose_port(
    preferred: int = DEFAULT_PORT,
) -> int:
    with socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM,
    ) as probe:
        try:
            probe.bind(
                (
                    "127.0.0.1",
                    preferred,
                )
            )
            return preferred
        except OSError:
            pass

    with socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM,
    ) as probe:
        probe.bind(
            (
                "127.0.0.1",
                0,
            )
        )
        return int(
            probe.getsockname()[1]
        )


def wait_and_open_browser(
    url: str,
    timeout: float = 45.0,
) -> None:
    health_url = (
        f"{url}/health"
    )

    deadline = (
        time.time()
        + timeout
    )

    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                health_url,
                timeout=1.0,
            ) as response:

                if (
                    200
                    <= response.status
                    < 500
                ):
                    webbrowser.open(
                        url,
                        new=2,
                    )

                    write_log(
                        f"Opened browser: {url}"
                    )

                    return

        except Exception:
            time.sleep(0.25)

    write_log(
        "Server did not become ready "
        f"within {timeout:.0f} seconds."
    )


def main() -> int:
    try:
        prepare_windowed_stdio()

        bootstrap_runtime()
        load_external_env()

        runtime = runtime_root()

        os.environ[
            "HUMOR_RUNTIME_ROOT"
        ] = str(runtime)

        os.environ[
            "HUMOR_MODEL_PATH"
        ] = str(
            runtime
            / "models"
            / "humor_transformer"
        )

        os.environ[
            "HUMOR_FEEDBACK_DB"
        ] = str(
            runtime
            / "humor_feedback.db"
        )

        os.environ[
            "ALLOW_BOOTSTRAP_MODEL"
        ] = "false"

        os.environ.setdefault(
            "HF_HUB_OFFLINE",
            "1",
        )

        os.environ.setdefault(
            "TRANSFORMERS_OFFLINE",
            "1",
        )

        os.environ.setdefault(
            "TOKENIZERS_PARALLELISM",
            "false",
        )

        port = choose_port()

        url = (
            f"http://127.0.0.1:{port}"
        )

        write_log("=" * 60)
        write_log(
            "Starting Humor Bot Milestone 2"
        )
        write_log(
            f"EXE directory: {exe_dir()}"
        )
        write_log(
            f"Runtime directory: {runtime}"
        )
        write_log(
            f"Resource directory: {resource_root()}"
        )
        write_log(
            f"Local URL: {url}"
        )

        from app import app as fastapi_app
        import uvicorn

        threading.Thread(
            target=wait_and_open_browser,
            args=(url,),
            daemon=True,
        ).start()

        uvicorn.run(
            fastapi_app,
            host="127.0.0.1",
            port=port,
            log_config=None,
            access_log=False,
        )

        return 0

    except KeyboardInterrupt:
        return 0

    except Exception as exc:
        import traceback

        write_log(
            traceback.format_exc()
        )

        show_error(
            "Humor Bot could not start.\n\n"
            f"{exc}\n\n"
            "See:\n"
            f"{log_path()}"
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(main())
