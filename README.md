# Otozu (音図) - Audio Spectrogram Conversion Library

Otozu (音図), combining the Japanese words for "sound" (音) and "diagram" (図), is a Python library that simplifies the conversion between audio files and spectrograms. Whether you're working on audio analysis, machine learning with sound data, or just curious about sound visualization, Otozu provides an intuitive interface for bidirectional conversion between audio files and their spectrogram representations.

## Why?

Ever needed to convert audio files into spectrograms? Maybe you're training a neural network for audio classification and want to understand what your model is actually learning. Otozu makes it easy to convert your audio dataset into spectrograms for training, but it goes further - you can also convert those learned representations back into audio. Want to know what features your CNN's first layer is detecting? Convert its activation patterns back into sound. Curious about what "maximum meowing" sounds like to your cat detector? Otozu can help you translate those neural patterns back into audio that humans can understand.

## Features

- Convert audio files to mel spectrograms (saved as PNG or NPY files)
- Reconstruct audio from spectrograms using the Griffin-Lim algorithm
- Support for multiple audio formats (WAV, MP3, FLAC, OGG)
- Batch processing of entire directories
- Command-line interface for easy usage
- Python API for integration into your projects
- Configurable spectrogram parameters
- Preserves metadata for accurate audio reconstruction

## Installation

```bash
pip install otozu
```

## Quick Start

### Command Line Usage

Convert an audio file to a spectrogram:

```bash
otozu audio-to-spec input.wav output_directory/
```

Convert all audio files in a directory:

```bash
otozu audio-to-spec input_directory/ output_directory/ --label dataset1
```

Reconstruct audio from a spectrogram:

```bash
otozu spec-to-audio spectrogram.png output_directory/
```

### Python API Usage

```python
from otozu import AudioSpectrogramConverter
from otozu.config import SpectrogramConfig

# Create a converter with custom settings
config = SpectrogramConfig(
    n_fft=2048,
    hop_length=512,
    n_mels=128
)
converter = AudioSpectrogramConverter(config)

# Convert audio to spectrogram
spec_path, metadata = converter.audio_to_spectrogram(
    "input.wav",
    "output_directory"
)

# Reconstruct audio from spectrogram
audio_path = converter.spectrogram_to_audio(
    "spectrogram.png",
    "reconstructed.wav"
)
```

## Advanced Configuration

The `SpectrogramConfig` class allows you to customize various aspects of the spectrogram generation:

```python
from otozu.config import SpectrogramConfig

config = SpectrogramConfig(
    n_fft=2048,          # FFT window size
    hop_length=512,      # Number of samples between successive frames
    n_mels=128,          # Number of mel bands
    sample_rate=44100,   # Target sample rate (None for original)
    center=False,        # Whether to pad signal at edges
    normalized_range=(0, 65535)  # Output range for normalization
)
```

## Technical Details

### Spectrogram Generation Process

1. Audio Loading: Files are loaded using librosa with configurable sample rate
2. Mel Spectrogram Creation: Converts audio to mel-scale spectrogram
3. Power to dB: Converts power spectrogram to decibel units
4. Normalization: Scales values to 16-bit range for PNG storage
5. Metadata Preservation: Stores original range and sample rate for reconstruction

### Audio Reconstruction Process

1. Spectrogram Loading: Loads normalized spectrogram data
2. Denormalization: Restores original decibel scale
3. Mel to Linear: Converts mel-scale spectrogram to linear-scale
4. Griffin-Lim Algorithm: Estimates phase information
5. Waveform Generation: Produces final audio output

## A Quick Note About Audio Reconstruction

When you convert a spectrogram back to audio, it won't sound exactly like the original. When we create spectrograms, we lose some information about the sound (specifically, the phase information). We do our best to guess what that missing information should be with the Griffin-Lim algorithm, but it's not perfect. The result is usually pretty good for most purposes, but it's going to be quite lossy.

## Contributing

Found a bug? Have an idea for a feature? PRs are always welcome! If you're thinking of adding something big, maybe open an issue first so we can chat about it.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

Otozu is basically just a friendly wrapper around several excellent libraries:

- librosa for audio processing
- numpy for numerical operations
- Pillow for image handling
- soundfile for audio file I/O
- typer for CLI interface

## Need Help?

If something's not working right or you're not sure how to do something, open an issue on GitHub. I'll do my best to help out when I can.

## Citation

If you use Otozu in your research, please cite:

```bibtex
@software{otozu2024,
  title = {Otozu: Audio Spectrogram Conversion Library},
  author = {Yamashiro, John},
  year = {2024},
  url = {https://github.com/jyaaan/otozu}
}
```

---

Built with 🎵 by an engineer who got tired of writing the same spectrogram conversion code over and over again. Bulk of code was taken from my sneeze detector, Kushami.
