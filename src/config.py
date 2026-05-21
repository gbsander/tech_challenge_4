from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ticker: str = "PETR4.SA"
    start_date: str = "2015-01-01"
    window: int = 60

    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    train_frac: float = 0.8
    val_frac: float = 0.1

    port: int = 8000
    log_level: str = "info"

    @property
    def model_path(self) -> Path:
        return ARTIFACTS_DIR / "model.keras"

    @property
    def scaler_path(self) -> Path:
        return ARTIFACTS_DIR / "scaler.pkl"

    @property
    def metadata_path(self) -> Path:
        return ARTIFACTS_DIR / "metadata.json"


settings = Settings()
