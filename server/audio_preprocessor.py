import io
import threading

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
    VAD_RMS_THRESHOLD = 0.005
    VAD_SPEECH_RATIO_THRESHOLD = 0.10


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


def detect_silence(
    audio: np.ndarray,
    sr: int,
    config: AudioConfig | None = None,
) -> tuple[bool, float, float]:
    if config is None:
        config = AudioConfig()

    if sr != config.SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=config.SAMPLE_RATE)

    max_samples = int(config.MAX_DURATION * config.SAMPLE_RATE)
    if len(audio) > max_samples:
        audio = audio[:max_samples]

    if np.max(np.abs(audio)) > 1.0:
        audio_norm = audio.astype(np.float32) / 32768.0
    else:
        audio_norm = audio.astype(np.float32)

    frame_len = config.HOP_LENGTH
    n_frames = len(audio_norm) // frame_len
    if n_frames == 0:
        return True, 0.0, 0.0

    voiced_count = 0
    rms_values = []
    for i in range(n_frames):
        frame = audio_norm[i * frame_len : (i + 1) * frame_len]
        rms = float(np.sqrt(np.mean(frame ** 2) + 1e-12))
        rms_values.append(rms)
        if rms >= config.VAD_RMS_THRESHOLD:
            voiced_count += 1

    avg_rms = float(np.mean(rms_values)) if rms_values else 0.0
    speech_ratio = voiced_count / n_frames
    is_silent = speech_ratio < config.VAD_SPEECH_RATIO_THRESHOLD
    return is_silent, speech_ratio, avg_rms


def detect_silence_from_wav_bytes(
    wav_bytes: bytes,
    config: AudioConfig | None = None,
) -> tuple[bool, float, float]:
    audio, sr = load_wav_bytes(wav_bytes)
    return detect_silence(audio, sr, config)

