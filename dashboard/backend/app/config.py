"""Small environment-based settings module for the backend."""

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = BACKEND_DIR.parents[1]

# Load only this backend's optional local file. Real credentials are never logged.
load_dotenv(BACKEND_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    db_host: str
    db_port: int
    db_user: str | None
    db_password: str | None
    db_name: str
    allowed_origins: tuple[str, ...]
    cpp_engine_path: Path
    simulation_timeout_seconds: float
    version: str = "0.1.0"


def get_settings() -> Settings:
    """Read settings on demand so current-session env changes take effect."""
    try:
        port = int(os.getenv("DB_PORT", "3306"))
    except ValueError:
        port = 3306
    try:
        timeout = float(os.getenv("SIMULATION_TIMEOUT_SECONDS", "30"))
    except ValueError:
        timeout = 30.0

    origin_text = os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    )
    origins = tuple(origin.strip() for origin in origin_text.split(",") if origin.strip())
    configured_engine = os.getenv("CPP_ENGINE_PATH")
    engine_path = (
        Path(configured_engine).expanduser()
        if configured_engine
        else REPOSITORY_ROOT / "core_engine" / "build" / "IntelligentResourceManager.exe"
    )
    if not engine_path.is_absolute():
        engine_path = BACKEND_DIR / engine_path

    return Settings(
        db_host=os.getenv("DB_HOST", "localhost"),
        db_port=port if 0 < port <= 65535 else 3306,
        db_user=os.getenv("DB_USER") or None,
        db_password=os.getenv("DB_PASSWORD"),
        db_name=os.getenv("DB_NAME", "resource_management_db"),
        allowed_origins=origins,
        cpp_engine_path=engine_path.resolve(),
        simulation_timeout_seconds=max(0.1, timeout),
    )
