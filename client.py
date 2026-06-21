#!/usr/bin/env python3
"""
语音情感分类 - CLI客户端

功能：
- 实时录制麦克风音频（每5秒一段）
- 调用服务端 /predict 接口进行情感分类
- 终端彩色打印情感概率分布（红=愤怒, 绿=开心, 蓝=难过, 灰=中性）
"""
import argparse
import io
import queue
import sys
import threading
import time
import wave
from datetime import datetime

import colorama
import numpy as np
import requests
import sounddevice as sd


SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2
CHUNK_DURATION = 5.0
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)


EMOTION_COLORS = {
    "happy": colorama.Fore.GREEN + colorama.Style.BRIGHT,
    "sad": colorama.Fore.BLUE + colorama.Style.BRIGHT,
    "angry": colorama.Fore.RED + colorama.Style.BRIGHT,
    "neutral": colorama.Fore.WHITE + colorama.Style.NORMAL,
}

EMOTION_ICONS = {
    "happy": "😊",
    "sad": "😢",
    "angry": "😠",
    "neutral": "😐",
}

EMOTION_ZH = {
    "happy": "开心",
    "sad": "难过",
    "angry": "愤怒",
    "neutral": "中性",
}


def encode_wav(audio_int16: np.ndarray, sr: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(sr)
        wf.writeframes(audio_int16.tobytes())
    return buf.getvalue()


def make_bar(pct: float, width: int = 25) -> str:
    filled = int(round(width * pct))
    filled = max(0, min(width, filled))
    bar = "█" * filled + "░" * (width - filled)
    return bar


def print_result(result: dict, index: int, latency_ms: float):
    emotions = result.get("emotions", {})
    dominant = result.get("dominant", "neutral")
    confidence = result.get("confidence", 0.0)

    timestamp = datetime.now().strftime("%H:%M:%S")
    dom_color = EMOTION_COLORS.get(dominant, "")
    dom_icon = EMOTION_ICONS.get(dominant, "?")
    dom_zh = EMOTION_ZH.get(dominant, dominant)
    reset = colorama.Style.RESET_ALL

    header = (
        f"{colorama.Fore.CYAN}[{timestamp}]{reset} "
        f"第{colorama.Fore.YELLOW}{index}{reset}段 "
        f"({latency_ms:.0f}ms) | "
        f"主导情绪: {dom_color}{dom_icon} {dom_zh} ({confidence:.1%}){reset}"
    )
    print(header)

    order = ["happy", "sad", "angry", "neutral"]
    for emo in order:
        prob = emotions.get(emo, 0.0)
        c = EMOTION_COLORS.get(emo, "")
        icon = EMOTION_ICONS.get(emo, "?")
        zh = EMOTION_ZH.get(emo, emo)
        bar = make_bar(prob)
        line = f"  {c}{icon} {zh:<4}{reset} {c}{bar}{reset} {prob:6.1%}"
        print(line)
    print()


class Recorder:
    def __init__(self, sr: int = SAMPLE_RATE, channels: int = CHANNELS):
        self.sr = sr
        self.channels = channels
        self._q: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: sd.InputStream | None = None
        self._buffer = np.array([], dtype=np.float32)
        self._lock = threading.Lock()

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"[录音状态] {status}", file=sys.stderr)
        with self._lock:
            self._buffer = np.concatenate([self._buffer, indata[:, 0].copy()])

    def start(self):
        self._stream = sd.InputStream(
            samplerate=self.sr,
            channels=self.channels,
            dtype="float32",
            blocksize=int(self.sr * 0.1),
            callback=self._callback,
        )
        self._stream.start()

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def get_chunk(self, n_samples: int) -> np.ndarray | None:
        with self._lock:
            if len(self._buffer) < n_samples:
                return None
            chunk = self._buffer[:n_samples].copy()
            self._buffer = self._buffer[n_samples:]
        return chunk


def send_wav(server_url: str, wav_bytes: bytes, timeout: float = 30.0) -> dict | None:
    endpoint = server_url.rstrip("/") + "/predict"
    try:
        files = {"file": ("chunk.wav", wav_bytes, "audio/wav")}
        resp = requests.post(endpoint, files=files, timeout=timeout)
        if resp.status_code != 200:
            print(
                f"{colorama.Fore.RED}[请求失败] HTTP {resp.status_code}: "
                f"{resp.json().get('detail', resp.text)}{colorama.Style.RESET_ALL}",
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


def run_realtime(server_url: str):
    if not check_server(server_url):
        print(
            f"{colorama.Fore.RED}[警告] 无法连接到服务端 {server_url}\n"
            f"请先启动服务端: python -m uvicorn main:app --reload --port 8000"
            f"{colorama.Style.RESET_ALL}"
        )

    colorama.init()
    recorder = Recorder()
    recorder.start()

    print(
        f"{colorama.Fore.MAGENTA}{colorama.Style.BRIGHT}"
        "====== 语音情感识别 (实时模式) ======\n"
        f"{colorama.Style.RESET_ALL}"
        f"{colorama.Fore.CYAN}采样率:{SAMPLE_RATE}Hz | 段长:{CHUNK_DURATION}秒 | 服务端:{server_url}\n"
        f"按 Ctrl+C 退出{colorama.Style.RESET_ALL}\n"
    )

    chunk_idx = 0
    try:
        while True:
            chunk = recorder.get_chunk(CHUNK_SAMPLES)
            if chunk is None:
                time.sleep(0.05)
                continue

            chunk_idx += 1
            audio_int16 = (chunk * 32767.0).astype(np.int16)
            wav_bytes = encode_wav(audio_int16)

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
            print(f"  [{i}] {dev['name']}{marker}")


def main():
    parser = argparse.ArgumentParser(
        description="语音情感分类 - CLI客户端",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 实时录音识别（每5秒一段）
  python client.py --server http://localhost:8000

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

    args = parser.parse_args()
    colorama.init()

    if args.list_devices:
        list_devices()
        return

    if args.device is not None:
        sd.default.device = (args.device, sd.default.device[1])

    if args.file:
        code = run_single_file(args.server, args.file)
        sys.exit(code)
    else:
        run_realtime(args.server)


if __name__ == "__main__":
    main()
