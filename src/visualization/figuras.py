"""Utilitários de caminho para figuras do projeto."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IMAGES = ROOT / "images"


def pasta(*partes: str) -> Path:
    return IMAGES.joinpath(*partes)
