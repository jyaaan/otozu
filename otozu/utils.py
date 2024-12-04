from enum import Enum
from pathlib import Path
from typing import List, Optional


class AudioFormat(Enum):
    WAV = ".wav"
    MP3 = ".mp3"
    FLAC = ".flac"
    OGG = ".ogg"


class SpectrogramFormat(Enum):
    PNG = ".png"
    NPY = ".npy"


def get_audio_files(
    directory: Path,
    formats: Optional[List[AudioFormat]] = None,
    sort: bool = True,
    as_iterator: bool = False,
) -> List[Path]:
    """Gat all audio files in directory matching specified format.

    Not specifying format will search for all supported file types."""
    if formats is None:
        formats = list(AudioFormat)

    if as_iterator:
        return (
            path for format in formats for path in directory.glob(f"*{format.value}")
        )

    files = []
    for format in formats:
        files.extend(directory.glob(f"*{format.value}"))

    return sorted(files) if sort else files


def ensure_directory(path: Path) -> Path:
    """Ensure directory exists, create if it doesn't."""
    path.mkdir(parents=True, exist_ok=True)
    return path
