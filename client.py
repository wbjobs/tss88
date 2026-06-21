#!/usr/bin/env python3
"""
语音情感分类 - CLI客户端

功能：
- 使用设备原生采样率录制麦克风音频，自动重采样到16kHz（服务端要求）
- 每5秒一段，客户端先做VAD静音检测，可跳过静音段
- 调用服务端 /predict 接口进行情感分类
- 终端彩色打印情感概率分布（红=愤怒, 绿=开心, 蓝=难过, 灰=中性, 深灰=静音）
"""
import argparse
import io
import sys
import threading
import time
import wave
from datetime import datetime

import colorama
import librosa
import numpy as np
import requests
import sounddevice as sd


TARGET_SAMPLE_RATE = 16000
RECORD_SAMPLE_RATES_TO_TRY = [48000, 44100, 32000, 22050, 16000]
CHANNELS = 1
SAMPLE_WIDTH = 2
CHUNK_DURATION = 5.0

VAD_RMS_THRESHOLD = 0.005
VAD_SPEECH_RATIO_THRESHOLD = 0.10
HOP_LENGTH = 256

SKIP_SILENT_ON_CLIENT = False


EMOTION_COLORS = {
    "happy": colorama.Fore.GREEN + colorama.Style.BRIGHT,
    "sad": colorama.Fore.BLUE + colorama.Style.BRIGHT,
    "angry": colorama.Fore.RED + colorama.Style.BRIGHT,
    "neutral": colorama.Fore.WHITE + colorama.Style.NORMAL,
    "silent": colorama.Fore.LIGHTBLACK_EX + colorama.Style.DIM,
}

EMOTION_ICONS = {
    "happy": "😊",
    "sad": "😢",
    "angry": "😠",
    "neutral": "😐",
    "silent": "🤫",
}

EMOTION_ZH = {
    "happy": "开心",
    "sad": "难过",
    "angry": "愤怒",
    "neutral": "中性",
    "silent": "静音",
}


def _detect_silence_local(audio: np.ndarray, sr: int) -> tuple[bool, float, float]:
    if sr != TARGET_SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)

    max_samples = int(CHUNK_DURATION * TARGET_SAMPLE_RATE)
    if len(audio) > max_samples:
        audio = audio[:max_samples]
    elif len(audio) < max_samples:
        audio = np.pad(audio, (0, max_samples - len(audio)), mode="constant")

    frame_len = HOP_LENGTH
    n_frames = len(audio) // frame_len
    if n_frames == 0:
        return True, 0.0, 0.0

    voiced_count = 0
    rms_values = []
    for i in range(n_frames):
        frame = audio[i * frame_len : (i + 1) * frame_len]
        rms = float(np.sqrt(np.mean(frame ** 2) + 1e-12))
        rms_values.append(rms)
        if rms >= VAD_RMS_THRESHOLD:
            voiced_count += 1

    avg_rms = float(np.mean(rms_values)) if rms_values else 0.0
    speech_ratio = voiced_count / n_frames
    is_silent = speech_ratio < VAD_SPEECH_RATIO_THRESHOLD
    return is_silent, speech_ratio, avg_rms


def encode_wav(audio_float32: np.ndarray, orig_sr: int) -> bytes:
    if orig_sr != TARGET_SAMPLE_RATE:
        audio_resampled = librosa.resample(
            audio_float32, orig_sr=orig_sr, target_sr=TARGET_SAMPLE_RATE
        )
    else:
        audio_resampled = audio_float32

    max_samples = int(CHUNK_DURATION * TARGET_SAMPLE_RATE)
    if len(audio_resampled) > max_samples:
        audio_resampled = audio_resampled[:max_samples]
    elif len(audio_resampled) < max_samples:
        audio_resampled = np.pad(
            audio_resampled,
            (0, max_samples - len(audio_resampled)),
            mode="constant",
        )

    audio_int16 = np.clip(audio_resampled, -1.0, 1.0)
    audio_int16 = (audio_int16 * 32767.0).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(TARGET_SAMPLE_RATE)
        wf.writeframes(audio_int16.tobytes())
    return buf.getvalue()


def make_bar(pct: float, width: int = 25) -> str:
    filled = int(round(width * pct))
    filled = max(0, min(width, filled))
    bar = "█" * filled + "░" * (width - filled)
    return bar


def print_result(result: dict, index: int, latency_ms: float, skipped: bool = False):
    emotions = result.get("emotions", {})
    dominant = result.get("dominant", "neutral")
    confidence = result.get("confidence", 0.0)
    speech_ratio = result.get("vad_speech_ratio", 0.0)
    avg_rms = result.get("vad_avg_rms", 0.0)

    timestamp = datetime.now().strftime("%H:%M:%S")
    dom_color = EMOTION_COLORS.get(dominant, "")
    dom_icon = EMOTION_ICONS.get(dominant, "?")
    dom_zh = EMOTION_ZH.get(dominant, dominant)
    reset = colorama.Style.RESET_ALL

    skip_tag = ""
    if skipped:
        skip_tag = f"{colorama.Fore.LIGHTBLACK_EX} [本地跳过]{reset}"

    header = (
        f"{colorama.Fore.CYAN}[{timestamp}]{reset} "
        f"第{colorama.Fore.YELLOW}{index}{reset}段 "
        f"({latency_ms:.0f}ms) "
        f"语音占比:{speech_ratio:.0%} RMS:{avg_rms:.4f}{skip_tag}"
        f"\n  主导情绪: {dom_color}{dom_icon} {dom_zh} ({confidence:.1%}){reset}"
    )
    print(header)

    order = ["happy", "sad", "angry", "neutral", "silent"]
    for emo in order:
        prob = emotions.get(emo, 0.0)
        c = EMOTION_COLORS.get(emo, "")
        icon = EMOTION_ICONS.get(emo, "?")
        zh = EMOTION_ZH.get(emo, emo)
        bar = make_bar(prob)
        line = f"  {c}{icon} {zh:<4}{reset} {c}{bar}{reset} {prob:6.1%}"
        print(line)
    print()


def _probe_record_sample_rate(device_id: int | None) -> int:
    if device_id is not None:
        dev_info = sd.query_devices(device_id)
        default_sr = int(dev_info["default_samplerate"])
    else:
        default_sr = int(sd.query_devices(sd.default.device[0])["default_samplerate"])

    candidates = []
    if default_sr > 0:
        candidates.append(default_sr)
    candidates.extend(RECORD_SAMPLE_RATES_TO_TRY)

    seen = set()
    for sr in candidates:
        if sr in seen:
            continue
        seen.add(sr)
        try:
            sd.check_input_settings(samplerate=sr, channels=CHANNELS, dtype="float32", device=device_id)
            return sr
        except Exception:
            continue
    raise RuntimeError(
        "无法找到设备支持的录音采样率，请用 --list-devices 查看并手动指定设备"
    )


class Recorder:
    def __init__(self, device_id: int | None, channels: int = CHANNELS):
        self.channels = channels
        self.device_id = device_id
        self.sr = _probe_record_sample_rate(device_id)

        self._stream: sd.InputStream | None = None
        self._buffer = np.array([], dtype=np.float32)
        self._lock = threading.Lock()

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(
                f"{colorama.Fore.YELLOW}[录音状态] {status}{colorama.Style.RESET_ALL}",
                file=sys.stderr,
            )
        with self._lock:
            self._buffer = np.concatenate([self._buffer, indata[:, 0].copy()])

    def start(self):
        blocksize = max(128, int(self.sr * 0.05))
        self._stream = sd.InputStream(
            samplerate=self.sr,
            channels=self.channels,
            dtype="float32",
            blocksize=blocksize,
            callback=self._callback,
            device=self.device_id,
        )
        self._stream.start()

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def get_chunk(self, duration_sec: float) -> tuple[np.ndarray, int] | None:
        n_samples = int(self.sr * duration_sec)
        with self._lock:
            if len(self._buffer) < n_samples:
                return None
            chunk = self._buffer[:n_samples].copy()
            self._buffer = self._buffer[n_samples:]
        return chunk, self.sr


def send_wav(server_url: str, wav_bytes: bytes, timeout: float = 30.0) -> dict | None:
    endpoint = server_url.rstrip("/") + "/predict"
    try:
        files = {"file": ("chunk.wav", wav_bytes, "audio/wav")}
        resp = requests.post(endpoint, files=files, timeout=timeout)
        if resp.status_code != 200:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            print(
                f"{colorama.Fore.RED}[请求失败] HTTP {resp.status_code}: "
                f"{detail}{colorama.Style.RESET_ALL}",
                file=sys.stderr,
            )
            return None
        return resp.json()
    except requests.ConnectionError:
        print(
            f"{colorama.Fore.RED}[连接错误] 无法连接到服务端 {server_url}{colorama.Style.RESET_ALL}",
            file=sys.stderr,
        )
        return None
    except requests.Timeout:
        print(
            f"{colorama.Fore.YELLOW}[超时] 服务端响应超时{colorama.Style.RESET_ALL}",
            file=sys.stderr,
        )
        return None
    except Exception as e:
        print(
            f"{colorama.Fore.RED}[错误] {e}{colorama.Style.RESET_ALL}",
            file=sys.stderr,
        )
        return None


def check_server(server_url: str) -> bool:
    try:
        resp = requests.get(server_url.rstrip("/") + "/health", timeout=5.0)
        return resp.status_code == 200
    except Exception:
        return False


def run_single_file(server_url: str, wav_path: str) -> int:
    print(f"{colorama.Fore.CYAN}分析文件: {wav_path}{colorama.Style.RESET_ALL}")
    try:
        with open(wav_path, "rb") as f:
            data = f.read()
    except FileNotFoundError:
        print(f"{colorama.Fore.RED}文件不存在: {wav_path}{colorama.Style.RESET_ALL}")
        return 1

    t0 = time.time()
    result = send_wav(server_url, data)
    latency = (time.time() - t0) * 1000
    if result is None:
        return 1
    print_result(result, 1, latency)
    return 0


def _build_local_silent_result(speech_ratio: float, avg_rms: float) -> dict:
    return {
        "emotions": {
            "happy": 0.0,
            "sad": 0.0,
            "angry": 0.0,
            "neutral": 0.0,
            "silent": 1.0,
        },
        "dominant": "silent",
        "confidence": 1.0,
        "is_silent": True,
        "vad_speech_ratio": speech_ratio,
        "vad_avg_rms": avg_rms,
    }


def run_realtime(server_url: str, device_id: int | None, skip_silent: bool):
    colorama.init()

    try:
        recorder = Recorder(device_id=device_id)
    except RuntimeError as e:
        print(f"{colorama.Fore.RED}{e}{colorama.Style.RESET_ALL}")
        return

    actual_sr = recorder.sr
    recorder.start()

    server_ok = check_server(server_url)
    if not server_ok:
        print(
            f"{colorama.Fore.YELLOW}[警告] 无法连接到服务端 {server_url}\n"
            f"请先启动服务端: python -m uvicorn main:app --reload --port 8000"
            f"{colorama.Style.RESET_ALL}\n"
        )

    print(
        f"{colorama.Fore.MAGENTA}{colorama.Style.BRIGHT}"
        "====== 语音情感识别 (实时模式) ======\n"
        f"{colorama.Style.RESET_ALL}"
        f"{colorama.Fore.CYAN}"
        f"录音设备采样率: {actual_sr}Hz (自动重采样到 {TARGET_SAMPLE_RATE}Hz)\n"
        f"段长: {CHUNK_DURATION}秒 | 服务端: {server_url}\n"
        f"VAD阈值: RMS>={VAD_RMS_THRESHOLD}, 语音占比>={VAD_SPEECH_RATIO_THRESHOLD:.0%}\n"
        f"客户端跳过静音: {'是' if skip_silent else '否'} | 按 Ctrl+C 退出"
        f"{colorama.Style.RESET_ALL}\n"
    )

    chunk_idx = 0
    try:
        while True:
            got = recorder.get_chunk(CHUNK_DURATION)
            if got is None:
                time.sleep(0.02)
                continue

            chunk_idx += 1
            audio_chunk, chunk_sr = got

            is_silent, speech_ratio, avg_rms = _detect_silence_local(audio_chunk, chunk_sr)

            if skip_silent and is_silent:
                fake_result = _build_local_silent_result(speech_ratio, avg_rms)
                print_result(fake_result, chunk_idx, 0.0, skipped=True)
                continue

            try:
                wav_bytes = encode_wav(audio_chunk, chunk_sr)
            except Exception as e:
                print(
                    f"{colorama.Fore.RED}[编码错误] {e}{colorama.Style.RESET_ALL}",
                    file=sys.stderr,
                )
                continue

            t0 = time.time()
            result = send_wav(server_url, wav_bytes)
            latency = (time.time() - t0) * 1000

            if result is not None:
                print_result(result, chunk_idx, latency)

    except KeyboardInterrupt:
        print(f"\n{colorama.Fore.YELLOW}用户中断，正在退出...{colorama.Style.RESET_ALL}")
    finally:
        recorder.stop()
        print(f"{colorama.Fore.GREEN}已停止录音{colorama.Style.RESET_ALL}")


def list_devices():
    print(colorama.Fore.CYAN + "可用音频输入设备:" + colorama.Style.RESET_ALL)
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            marker = " *" if i == sd.default.device[0] else ""
            sr = int(dev["default_samplerate"])
            print(f"  [{i}] {dev['name']} (默认采样率:{sr}Hz){marker}")


def main():
    parser = argparse.ArgumentParser(
        description="语音情感分类 - CLI客户端",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 实时录音识别（每5秒一段，默认不跳过静音）
  python client.py --server http://localhost:8000

  # 实时录音并在客户端跳过静音段（不发送请求）
  python client.py --server http://localhost:8000 --skip-silent

  # 识别单个WAV文件
  python client.py --server http://localhost:8000 --file test.wav

  # 列出可用麦克风
  python client.py --list-devices
""",
    )
    parser.add_argument(
        "--server",
        default="http://localhost:8000",
        help="服务端地址 (默认: http://localhost:8000)",
    )
    parser.add_argument("--file", help="指定单个WAV文件进行分析（非实时模式）")
    parser.add_argument("--list-devices", action="store_true", help="列出可用音频输入设备")
    parser.add_argument("--device", type=int, help="指定输入设备ID（使用--list-devices查看）")
    parser.add_argument(
        "--skip-silent",
        action="store_true",
        help="客户端检测到静音时跳过发送请求（仅本地打印静音结果）",
    )

    args = parser.parse_args()
    colorama.init()

    if args.list_devices:
        list_devices()
        return

    if args.file:
        code = run_single_file(args.server, args.file)
        sys.exit(code)
    else:
        run_realtime(args.server, args.device, args.skip_silent)


if __name__ == "__main__":
    main()
