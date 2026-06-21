"""
滑动窗口情感推理管理器

每个WebSocket连接对应一个SlidingWindowSession实例，负责：
1. 接收客户端每500ms发来的16kHz PCM音频片段
2. 累积维护一个5秒的滑动音频窗口
3. 按指定步长触发推理（默认每500ms推理一次，窗口滑动500ms）
4. 执行VAD静音检测，仅在有语音时进行ONNX推理
"""
import asyncio
import io
import threading
import time
import wave
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from . import (
    AudioConfig,
    EmotionModel,
    SILENT_LABEL,
    detect_silence,
    preprocess_audio,
)


TARGET_SAMPLE_RATE = 16000
WINDOW_DURATION = 5.0
WINDOW_SAMPLES = int(TARGET_SAMPLE_RATE * WINDOW_DURATION)
INFERENCE_INTERVAL = 0.5
SILENT_RESULT = {
    "happy": 0.0,
    "sad": 0.0,
    "angry": 0.0,
    "neutral": 0.0,
    "silent": 1.0,
}


def _encode_wav_for_preprocess(audio_float32: np.ndarray, sr: int) -> bytes:
    audio_int16 = np.clip(audio_float32, -1.0, 1.0)
    audio_int16 = (audio_int16 * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio_int16.tobytes())
    return buf.getvalue()


@dataclass
class EmotionFrame:
    timestamp: float
    emotions: dict[str, float]
    dominant: str
    confidence: float
    is_silent: bool
    vad_speech_ratio: float
    vad_avg_rms: float
    inference_ms: float = 0.0


class SlidingWindowSession:
    def __init__(
        self,
        model: EmotionModel | None,
        config: AudioConfig | None = None,
        window_duration: float = WINDOW_DURATION,
        inference_interval: float = INFERENCE_INTERVAL,
    ):
        if config is None:
            config = AudioConfig()
        self.config = config
        self.model = model
        self.window_samples = int(window_duration * config.SAMPLE_RATE)
        self.inference_interval = inference_interval

        self._buffer_lock = threading.Lock()
        self._inference_lock = threading.Lock()

        self._audio_buffer = np.array([], dtype=np.float32)

        self._result_queue: asyncio.Queue[EmotionFrame] = asyncio.Queue()
        self._inference_task: asyncio.Task | None = None
        self._running = False
        self._last_inference_time: float = 0.0
        self._frame_counter: int = 0
        self._inference_counter: int = 0

        self._recent_results: deque[EmotionFrame] = deque(maxlen=100)

    async def start(self):
        self._running = True
        self._inference_task = asyncio.create_task(self._inference_loop())

    async def stop(self):
        self._running = False
        if self._inference_task is not None and not self._inference_task.done():
            self._inference_task.cancel()
            try:
                await self._inference_task
            except asyncio.CancelledError:
                pass
            self._inference_task = None

    def is_running(self) -> bool:
        return self._running

    def feed_audio(self, pcm_bytes: bytes, sample_rate: int):
        with self._buffer_lock:
            audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)

            if sample_rate != self.config.SAMPLE_RATE:
                import librosa
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=self.config.SAMPLE_RATE)

            self._audio_buffer = np.concatenate([self._audio_buffer, audio])

            if len(self._audio_buffer) > self.window_samples * 3:
                self._audio_buffer = self._audio_buffer[-self.window_samples * 2 :]

        self._frame_counter += 1

    def _get_window(self) -> np.ndarray | None:
        with self._buffer_lock:
            n = len(self._audio_buffer)
            if n < self.window_samples:
                return None
            return self._audio_buffer[-self.window_samples:].copy()

    def _slide_window(self, slide_seconds: float):
        slide_samples = int(slide_seconds * self.config.SAMPLE_RATE)
        with self._buffer_lock:
            if len(self._audio_buffer) > slide_samples:
                self._audio_buffer = self._audio_buffer[slide_samples:]

    async def _inference_loop(self):
        try:
            while self._running:
                await asyncio.sleep(self.inference_interval)

                window = self._get_window()
                if window is None:
                    continue

                t0 = time.perf_counter()

                is_silent, speech_ratio, avg_rms = detect_silence(
                    window, self.config.SAMPLE_RATE, self.config
                )

                if is_silent:
                    emotions = dict(SILENT_RESULT)
                    dominant = SILENT_LABEL
                    confidence = 1.0
                else:
                    if self.model is None:
                        emotions = dict(SILENT_RESULT)
                        dominant = SILENT_LABEL
                        confidence = 1.0
                    else:
                        try:
                            wav_bytes = _encode_wav_for_preprocess(
                                window, self.config.SAMPLE_RATE
                            )
                            mel_spec = preprocess_audio(wav_bytes, self.config)
                            with self._inference_lock:
                                emotions = self.model.predict(mel_spec)
                            emotions[SILENT_LABEL] = 0.0
                        except Exception:
                            emotions = dict(SILENT_RESULT)
                            emotions[SILENT_LABEL] = 0.5
                            emotions["neutral"] = 0.5

                        dominant = max(emotions, key=emotions.get)
                        confidence = emotions[dominant]

                inference_ms = (time.perf_counter() - t0) * 1000

                frame = EmotionFrame(
                    timestamp=time.time(),
                    emotions=emotions,
                    dominant=dominant,
                    confidence=confidence,
                    is_silent=is_silent,
                    vad_speech_ratio=speech_ratio,
                    vad_avg_rms=avg_rms,
                    inference_ms=inference_ms,
                )

                self._inference_counter += 1
                self._last_inference_time = time.time()
                self._recent_results.append(frame)

                try:
                    self._result_queue.put_nowait(frame)
                except asyncio.QueueFull:
                    try:
                        self._result_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                    self._result_queue.put_nowait(frame)

        except asyncio.CancelledError:
            pass
        except Exception:
            self._running = False

    async def get_next_result(self, timeout: float | None = None) -> EmotionFrame | None:
        try:
            return await asyncio.wait_for(self._result_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    def get_stats(self) -> dict:
        return {
            "audio_frames_received": self._frame_counter,
            "inferences_performed": self._inference_counter,
            "buffer_samples": len(self._audio_buffer),
            "buffer_seconds": len(self._audio_buffer) / self.config.SAMPLE_RATE,
            "last_inference_ts": self._last_inference_time,
        }
