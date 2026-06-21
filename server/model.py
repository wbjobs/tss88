import os
import threading
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .audio_preprocessor import AudioConfig, get_mel_spec_shape


EMOTION_LABELS = ["happy", "sad", "angry", "neutral"]
ALL_EMOTION_LABELS = EMOTION_LABELS + ["silent"]
SILENT_LABEL = "silent"


class EmotionModel:
    def __init__(self, model_path: str | os.PathLike, config: AudioConfig | None = None):
        if config is None:
            config = AudioConfig()
        self.config = config

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        providers = ["CPUExecutionProvider"]
        try:
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            elif "DmlExecutionProvider" in available:
                providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        except Exception:
            pass

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 0
        sess_options.inter_op_num_threads = 0
        sess_options.log_severity_level = 3

        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=providers,
        )

        self._lock = threading.Lock()

        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_name = self.session.get_outputs()[0].name

        expected_shape = get_mel_spec_shape(config)
        self.expected_height = expected_shape[0]
        self.expected_width = expected_shape[1]

    def predict(self, mel_spec: np.ndarray) -> dict[str, float]:
        if mel_spec.ndim == 2:
            mel_spec = mel_spec[np.newaxis, np.newaxis, ...]
        elif mel_spec.ndim == 3:
            mel_spec = mel_spec[np.newaxis, ...]

        mel_spec = mel_spec.astype(np.float32)

        with self._lock:
            outputs = self.session.run([self.output_name], {self.input_name: mel_spec})
        logits = outputs[0]

        if logits.shape[-1] != len(EMOTION_LABELS):
            raise ValueError(
                f"Model output has {logits.shape[-1]} classes, "
                f"but expected {len(EMOTION_LABELS)}: {EMOTION_LABELS}"
            )

        probs = self._softmax(logits)
        probs = probs.reshape(-1)

        return {label: float(probs[i]) for i, label in enumerate(EMOTION_LABELS)}

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x_max = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
