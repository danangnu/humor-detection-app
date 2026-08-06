from __future__ import annotations
import ctypes, os, socket, sys, threading, time, urllib.request, webbrowser
from pathlib import Path

APP_NAME = "Humor Bot"
DEFAULT_PORT = 8000

def exe_dir():
    return Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent

def log_path():
    return exe_dir() / "HumorBot.log"

def write_log(message):
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        with log_path().open("a", encoding="utf-8") as f:
            f.write(f"[{stamp}] {message}\n")
    except Exception:
        pass

def show_error(message):
    write_log("ERROR: " + message)
    if os.name == "nt":
        try:
            ctypes.windll.user32.MessageBoxW(0, message, APP_NAME, 0x10)
            return
        except Exception:
            pass

def prepare_windowed_stdio():
    try:
        if sys.stdout is None:
            sys.stdout = open(os.devnull, "w", encoding="utf-8")
        if sys.stderr is None:
            sys.stderr = open(os.devnull, "w", encoding="utf-8")
    except Exception:
        pass

def load_external_env():
    env_file = exe_dir() / ".env"
    if env_file.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file, override=False)
            write_log(f"Loaded environment file: {env_file}")
        except Exception as exc:
            write_log(f"Could not load .env: {exc}")

def choose_port(preferred=DEFAULT_PORT):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])

def wait_and_open_browser(url, timeout=30):
    health = url + "/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health, timeout=1) as r:
                if 200 <= r.status < 500:
                    webbrowser.open(url, new=2)
                    write_log("Opened browser: " + url)
                    return
        except Exception:
            time.sleep(0.25)

def main():
    try:
        prepare_windowed_stdio()
        load_external_env()

        os.environ["ALLOW_BOOTSTRAP_MODEL"] = "false"
        os.environ.setdefault("HUMOR_MODEL_PATH", "models/humor_transformer")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        port = choose_port()
        url = f"http://127.0.0.1:{port}"
        write_log("=" * 60)
        write_log("Starting Humor Bot")
        write_log("Local URL: " + url)

        from app import app as fastapi_app
        import uvicorn

        threading.Thread(target=wait_and_open_browser, args=(url,), daemon=True).start()

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
        write_log(traceback.format_exc())
        show_error(f"Humor Bot could not start.\n\n{exc}\n\nSee {log_path()} for details.")
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
