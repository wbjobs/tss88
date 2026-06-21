import io
import numpy as np
import librosa
import soundfile as sf


class AudioConfig:
    SAMPLE_RATE = 16000
    MAX_DURATION = 5.0
    N_MELS = 128
    N_FFT = 512
    HOP_LENGTH = 256
    WIN_LENGTH = 512
    FMIN = 20
    FMAX = 8000


def load_wav_bytes(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(io.BytesIO(wav_bytes))
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio, sr


def preprocess_audio(
    wav_bytes: bytes,
    config: AudioConfig | None = None,
) -> np.ndarray:
    if config is None:
        config = AudioConfig()

    audio, sr = load_wav_bytes(wav_bytes)

    if sr != config.SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=config.SAMPLE_RATE)

    max_samples = int(config.MAX_DURATION * config.SAMPLE_RATE)
    if len(audio) > max_samples:
        audio = audio[:max_samples]
    elif len(audio) < max_samples:
        padding = np.zeros(max_samples - len(audio), dtype=audio.dtype)
        audio = np.concatenate([audio, padding])

    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=config.SAMPLE_RATE,
        n_mels=config.N_MELS,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        win_length=config.WIN_LENGTH,
        fmin=config.FMIN,
        fmax=config.FMAX,
        power=2.0,
    )

    log_mel = librosa.power_to_db(mel_spec, ref=np.max)

    eps = 1e-10
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + eps)

    return log_mel.astype(np.float32)


def get_mel_spec_shape(config: AudioConfig | None = None) -> tuple[int, int]:
    if config is None:
        config = AudioConfig()
    max_samples = int(config.MAX_DURATION * config.SAMPLE_RATE)
    n_frames = int(np.ceil(max_samples / config.HOP_LENGTH))
    return (config.N_MELS, n_frames)
