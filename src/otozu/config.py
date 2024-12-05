from dataclasses import dataclass
from typing import Optional


@dataclass
class SpectrogramConfig:
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    sample_rate: Optional[int] = None
    center: bool = False
    normalized_range: tuple[float, float] = (0, 65535)  # 16-bit range
