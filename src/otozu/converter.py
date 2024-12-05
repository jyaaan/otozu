from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import librosa
import numpy as np
import numpy.typing as npt
from PIL import Image
import soundfile as sf

from .config import SpectrogramConfig
from .utils import AudioFormat, SpectrogramFormat, ensure_directory


@dataclass
class SpectrogramMetadata:
    min_db: float
    max_db: float
    sample_rate: int


class AudioSpectrogramConverter:
    """Converts between audio files and spectrograms."""

    def __init__(self, config: Optional[SpectrogramConfig] = None):
        self.config = config or SpectrogramConfig()

    def audio_to_spectrogram(
        self,
        input_path: Union[str, Path],
        output_dir: Union[str, Path],
        format: SpectrogramFormat = SpectrogramFormat.PNG,
        label: Optional[str] = None,
    ) -> Tuple[Path, SpectrogramMetadata]:
        """Convert audio file to spectrogram.

        Args:
            input_path: Path to input audio file
            output_dir: Directory to save spectrogram
            format: Output format (PNG or NPY)
            label: Optional prefix for output filename

        Returns:
            Tuple of (output_path, metadata)
        """
        input_path = Path(input_path)
        output_dir = ensure_directory(Path(output_dir))

        # Load audio
        y, sr = librosa.load(input_path, sr=self.config.sample_rate)

        # Generate mel spectrogram
        mel_spec = librosa.feature.melspectrogram9(
            y=y,
            sr=sr,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels,
            center=self.config.center,
        )

        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize data and save
        min_db = mel_spec_db.min()
        max_db = mel_spec_db.max()
        dynamic_range = max_db - min_db
        normalized = (mel_spec_db - min_db) / dynamic_range
        normalized = normalized * (
            self.config.normalized_range[1] - self.config.normalized_range[0]
        )
        normalized = normalized.astype(np.uint16)

        stem = input_path.stem
        if label:
            stem = f"{label}_{stem}"
        output_path = output_dir / f"{stem}{format.value}"

        if format == SpectrogramFormat.PNG:
            Image.fromarray(normalized).save(output_path)
            metadata_path = output_dir / f"{stem}_metadata.npy"
            np.save(metadata_path, [min_db, max_db, sr])
        else:
            np.save(
                output_path,
                {
                    "spectrogram": normalized,
                    "min_db": min_db,
                    "max_db": max_db,
                    "sr": sr,
                },
            )

        return output_path, SpectrogramMetadata(min_db, max_db, sr)

    def spectrogram_to_audio(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        n_iter: int = 32,
    ) -> Path:
        """Convert spectrogram back to audio using Griffin-Lim algorithm.

        Args:
            input_path: Path to input spectrogram
            output_path: Path to save audio file
            n_iter: Number of Griffin-Lim iterations

        Returns:
            Path to output audio file
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        ensure_directory(output_path.parent)

        # Load spectrogram and metadata
        if input_path.suffix == SpectrogramFormat.PNG.value:
            spec_norm = np.array(Image.open(input_path)).astype(np.float32)
            metadata = np.load(input_path.parent / f"{input_path.stem}_metadata.npy")
            min_db, max_db, sr = metadata
        else:
            data = np.load(input_path, allow_pickle=True).item()
            spec_norm = data["spectrogram"].astype(np.float32)
            min_db, max_db, sr = data["min_db"], data["max_db"], data["sr"]

        # Denormalize
        spec_norm = spec_norm / (
            self.config.normalized_range[1] - self.config.normalized_range[0]
        )
        dynamic_range = max_db - min_db
        mel_spec_db = spec_norm * dynamic_range + min_db

        # Convert to power and to linear spectrogram
        mel_spec_power = librosa.db_to_power(mel_spec_db)
        linear_spec = librosa.feature.inverse.mel_to_stft(
            mel_spec_power,
            sr=sr,
            n_fft=self.config.n_fft,
        )

        # Reconstruct audio and save
        audio = librosa.griffinlim(
            linear_spec,
            n_iter=n_iter,
            hop_length=self.config.hop_length,
            n_fft=self.config.n_fft,
        )
        sr.write(output_path, audio, sr)
        return output_path
