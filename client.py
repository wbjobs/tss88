#!/usr/bin/env python3
"""
语音情感分类 - CLI客户端 (WebSocket 流式版)

功能：
- 使用设备原生采样率录制麦克风音频，自动重采样到16kHz
- 建立WebSocket长连接，每500ms发送一小段PCM音频流
- 服务端实时累积5秒滑窗，每500ms推回一次情感推理结果
- 终端实时显示当前情感概率，以及最近10个窗口的ASCII柱状图历史曲线
"""
import argparse
import asyncio
import json
import queue
import sys
import threading
import time
from collections import deque
from datetime import datetime

import colorama
import librosa
import numpy as np
import sounddevice as sd
import websockets


TARGET_SAMPLE_RATE = 16000
RECORD_SAMPLE_RATES_TO_TRY = [48000, 44100, 32000, 22050, 16000]
CHANNELS = 1
CHUNK_DURATION = 0.5
HISTORY_WINDOW = 10

VAD_RMS_THRESHOLD = 0.005
VAD_SPEECH_RATIO_THRESHOLD = 0.10
HOP_LENGTH = 256


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

EMOTION_ORDER = ["happy", "sad", "angry", "neutral", "silent"]


def make_bar(pct: float, width: int = 20) -> str:
    filled = int(round(width * pct))
    filled = max(0, min(width, filled))
    bar = "█" * filled + "░" * (width - filled)
    return bar


def make_histogram(history: deque[dict], height: int = 8) -> str:
    if len(history) == 0:
        return "  (等待数据...)\n"

    n = len(history)
    lines = []

    for emo in EMOTION_ORDER:
        probs = [frame["emotions"].get(emo, 0.0) for frame in history]
        max_p = max(max(probs), 0.01)

        row_chars = [" " * height for _ in range(height)]

        for t_idx, p in enumerate(probs):
            bar_h = int(round(height * p / max_p)) if max_p > 0 else 0
            bar_h = max(0, min(height, bar_h))
            for h in range(height):
                row_idx = height - 1 - h
                pos = t_idx * 2
                if h < bar_h:
                    s = row_chars[row_idx]
                    c = "█"
                    row_chars[row_idx] = s[:pos] + c + s[pos + 1 :]

        c = EMOTION_COLORS.get(emo, "")
        icon = EMOTION_ICONS.get(emo, "?")
        zh = EMOTION_ZH.get(emo, emo)
        reset = colorama.Style.RESET_ALL

        if max_p < 0.01:
            lines.append(f"  {c}{icon} {zh:<4}{reset}   (无数据)")
            continue

        lines.append(f"  {c}{icon} {zh:<4}{reset}  {c}{row_chars[0]}{reset}")
        for r in row_chars[1:]:
            lines.append(f"         {c}{r}{reset}")
        lines.append(f"         {c}{'─' * (n * 2 - 1)}{reset}  max={max_p:.0%}")
        lines.append("")

    x_labels = "  "
    for i in range(n):
        if i % 2 == 0:
            idx = len(history) - n + i
            seq = history[idx].get("seq", idx + 1)
            label = f"#{seq:<3}"
        else:
            label = "   "
        x_labels += label[:2]
    lines.append(x_labels)
    lines.append(f"  最近{n}个推理窗口 (每500ms一个)")
    return "\n".join(lines) + "\n"


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


def encode_pcm(audio_float32: np.ndarray, orig_sr: int) -> bytes:
    if orig_sr != TARGET_SAMPLE_RATE:
        audio_resampled = librosa.resample(
            audio_float32, orig_sr=orig_sr, target_sr=TARGET_SAMPLE_RATE
        )
    else:
        audio_resampled = audio_float32

    audio_int16 = np.clip(audio_resampled, -1.0, 1.0)
    audio_int16 = (audio_int16 * 32767.0).astype(np.int16)
    return audio_int16.tobytes()


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
            sd.check_input_settings(
                samplerate=sr, channels=CHANNELS, dtype="float32", device=device_id
            )
            return sr
        except Exception:
            continue
    raise RuntimeError(
        "无法找到设备支持的录音采样率，请用 --list-devices 查看并手动指定设备"
    )


class Recorder:
    def __init__(self, device_id: int | None, chunk_duration: float = CHUNK_DURATION):
        self.chunk_duration = chunk_duration
        self.device_id = device_id
        self.sr = _probe_record_sample_rate(device_id)

        self._stream: sd.InputStream | None = None
        self._buffer = np.array([], dtype=np.float32)
        self._lock = threading.Lock()
        self._chunk_samples = int(self.sr * self.chunk_duration)

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
            channels=CHANNELS,
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

    def get_chunk(self) -> tuple[np.ndarray, int] | None:
        with self._lock:
            if len(self._buffer) < self._chunk_samples:
                return None
            chunk = self._buffer[: self._chunk_samples].copy()
            self._buffer = self._buffer[self._chunk_samples :]
        return chunk, self.sr


def list_devices():
    print(colorama.Fore.CYAN + "可用音频输入设备:" + colorama.Style.RESET_ALL)
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            marker = " *" if i == sd.default.device[0] else ""
            sr = int(dev["default_samplerate"])
            print(f"  [{i}] {dev['name']} (默认采样率:{sr}Hz){marker}")


def print_header(record_sr: int, ws_url: str, mode: str):
    print(
        f"{colorama.Fore.MAGENTA}{colorama.Style.BRIGHT}"
        "====== 语音情感识别 (实时流模式) ======\n"
        f"{colorama.Style.RESET_ALL}"
        f"{colorama.Fore.CYAN}"
        f"模式: {mode}\n"
        f"录音设备采样率: {record_sr}Hz (自动重采样到 {TARGET_SAMPLE_RATE}Hz)\n"
        f"帧长: {CHUNK_DURATION * 1000:.0f}ms | 滑窗: 5s | 推理频率: 2Hz\n"
        f"服务端: {ws_url}\n"
        f"VAD阈值: RMS>={VAD_RMS_THRESHOLD}, 语音占比>={VAD_SPEECH_RATIO_THRESHOLD:.0%}\n"
        f"按 Ctrl+C 退出"
        f"{colorama.Style.RESET_ALL}\n"
    )


def render_display(
    result: dict,
    history: deque[dict],
    chunks_sent: int,
    results_received: int,
    local_vad_silent: bool,
    local_vad_ratio: float,
    local_vad_rms: float,
):
    seq = result.get("seq", 0)
    emotions = result.get("emotions", {})
    dominant = result.get("dominant", "neutral")
    confidence = result.get("confidence", 0.0)
    is_silent = result.get("is_silent", False)
    speech_ratio = result.get("vad_speech_ratio", 0.0)
    avg_rms = result.get("vad_avg_rms", 0.0)
    inference_ms = result.get("inference_ms", 0.0)

    timestamp = datetime.now().strftime("%H:%M:%S")
    dom_color = EMOTION_COLORS.get(dominant, "")
    dom_icon = EMOTION_ICONS.get(dominant, "?")
    dom_zh = EMOTION_ZH.get(dominant, dominant)
    reset = colorama.Style.RESET_ALL

    header = (
        f"{colorama.Fore.CYAN}[{timestamp}]{reset} "
        f"帧:{colorama.Fore.YELLOW}#{seq}{reset} "
        f"发送:{chunks_sent} 接收:{results_received} | "
        f"推理耗时:{inference_ms:.0f}ms\n"
        f"  本地VAD: {'静音' if local_vad_silent else '有语音'} "
        f"语音占比:{local_vad_ratio:.0%} RMS:{local_vad_rms:.4f}\n"
        f"  服务端VAD: {'静音' if is_silent else '有语音'} "
        f"语音占比:{speech_ratio:.0%} RMS:{avg_rms:.4f}\n"
        f"  主导情绪: {dom_color}{dom_icon} {dom_zh} ({confidence:.1%}){reset}"
    )
    print(header)

    for emo in EMOTION_ORDER:
        prob = emotions.get(emo, 0.0)
        c = EMOTION_COLORS.get(emo, "")
        icon = EMOTION_ICONS.get(emo, "?")
        zh = EMOTION_ZH.get(emo, emo)
        bar = make_bar(prob, width=25)
        line = f"  {c}{icon} {zh:<4}{reset} {c}{bar}{reset} {prob:6.1%}"
        print(line)

    print()
    print(f"{colorama.Fore.MAGENTA}====== 最近{HISTORY_WINDOW}个窗口历史曲线 ======{reset}")
    print(make_histogram(history))
    print("\033[F" * 0, end="")


async def _ws_receiver(ws, result_queue: asyncio.Queue, stop_event: asyncio.Event):
    try:
        async for msg in ws:
            if stop_event.is_set():
                break
            if isinstance(msg, str):
                try:
                    data = json.loads(msg)
                except json.JSONDecodeError:
                    continue
                await result_queue.put(data)
    except websockets.exceptions.ConnectionClosed:
        pass
    except Exception as e:
        print(
            f"\n{colorama.Fore.RED}[WebSocket错误] {e}{colorama.Style.RESET_ALL}",
            file=sys.stderr,
        )
    finally:
        stop_event.set()


async def run_stream_client(
    server_url: str,
    device_id: int | None,
    use_http: bool = False,
    history_size: int = HISTORY_WINDOW,
):
    colorama.init()

    try:
        recorder = Recorder(device_id=device_id)
    except RuntimeError as e:
        print(f"{colorama.Fore.RED}{e}{colorama.Style.RESET_ALL}")
        return

    actual_sr = recorder.sr
    recorder.start()

    mode = "HTTP 轮询" if use_http else "WebSocket 流式"
    ws_url = server_url.replace("http://", "ws://").replace("https://", "wss://") + "/ws/stream"
    print_header(actual_sr, ws_url, mode)

    result_queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=20)
    stop_event = asyncio.Event()

    history: deque[dict] = deque(maxlen=history_size)

    ws = None
    receiver_task = None

    if not use_http:
        try:
            ws = await websockets.connect(ws_url, ping_interval=30, ping_timeout=10)
            hello = await asyncio.wait_for(ws.recv(), timeout=5.0)
            if isinstance(hello, str):
                hello_data = json.loads(hello)
                if hello_data.get("type") == "hello":
                    print(
                        f"{colorama.Fore.GREEN}[WebSocket] 已连接 | "
                        f"模型已加载: {hello_data.get('model_loaded', False)} | "
                        f"服务端窗口: {hello_data.get('window_duration', 5)}s | "
                        f"推理频率: {1.0 / hello_data.get('inference_interval', 0.5):.0f}Hz"
                        f"{colorama.Style.RESET_ALL}\n"
                    )
        except asyncio.TimeoutError:
            print(
                f"{colorama.Fore.RED}[WebSocket] 服务端响应超时{colorama.Style.RESET_ALL}"
            )
            recorder.stop()
            return
        except Exception as e:
            print(
                f"{colorama.Fore.RED}[WebSocket] 连接失败: {e}\n"
                f"请确认服务端已启动: python -m uvicorn main:app --port 8000"
                f"{colorama.Style.RESET_ALL}"
            )
            recorder.stop()
            return

        receiver_task = asyncio.create_task(
            _ws_receiver(ws, result_queue, stop_event)
        )

    chunks_sent = 0
    results_received = 0
    seq_counter = 0
    local_vad_silent = False
    local_vad_ratio = 0.0
    local_vad_rms = 0.0

    try:
        while not stop_event.is_set():
            chunk = recorder.get_chunk()

            if chunk is None:
                await asyncio.sleep(0.02)
                continue

            chunks_sent += 1
            audio_chunk, chunk_sr = chunk

            is_silent, speech_ratio, avg_rms = _detect_silence_local(audio_chunk, chunk_sr)
            local_vad_silent = is_silent
            local_vad_ratio = speech_ratio
            local_vad_rms = avg_rms

            try:
                pcm_bytes = encode_pcm(audio_chunk, chunk_sr)
            except Exception as e:
                print(
                    f"\n{colorama.Fore.RED}[编码错误] {e}{colorama.Style.RESET_ALL}",
                    file=sys.stderr,
                )
                await asyncio.sleep(0.01)
                continue

            if not use_http and ws is not None:
                try:
                    await ws.send(pcm_bytes)
                except Exception:
                    print(
                        f"\n{colorama.Fore.RED}[WebSocket] 连接断开{colorama.Style.RESET_ALL}",
                        file=sys.stderr,
                    )
                    break

            while not result_queue.empty():
                try:
                    result = result_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

                if result.get("type") == "result":
                    results_received += 1
                    seq_counter += 1
                    result["seq"] = result.get("seq", seq_counter)
                    history.append(result)

                    print("\033c", end="")
                    print_header(actual_sr, ws_url, mode)
                    render_display(
                        result,
                        history,
                        chunks_sent,
                        results_received,
                        local_vad_silent,
                        local_vad_ratio,
                        local_vad_rms,
                    )

            if use_http:
                pass

            await asyncio.sleep(0.01)

    except KeyboardInterrupt:
        print(f"\n{colorama.Fore.YELLOW}用户中断，正在退出...{colorama.Style.RESET_ALL}")
        stop_event.set()
    finally:
        recorder.stop()
        if receiver_task is not None and not receiver_task.done():
            receiver_task.cancel()
            try:
                await receiver_task
            except Exception:
                pass
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass
        print(f"{colorama.Fore.GREEN}已停止录音{colorama.Style.RESET_ALL}")


def run_http_client(server_url: str, wav_path: str) -> int:
    import requests

    print(f"{colorama.Fore.CYAN}分析文件: {wav_path}{colorama.Style.RESET_ALL}")
    try:
        with open(wav_path, "rb") as f:
            data = f.read()
    except FileNotFoundError:
        print(f"{colorama.Fore.RED}文件不存在: {wav_path}{colorama.Style.RESET_ALL}")
        return 1

    t0 = time.time()
    endpoint = server_url.rstrip("/") + "/predict"
    try:
        files = {"file": ("chunk.wav", data, "audio/wav")}
        resp = requests.post(endpoint, files=files, timeout=30)
        latency = (time.time() - t0) * 1000
        if resp.status_code != 200:
            print(
                f"{colorama.Fore.RED}[请求失败] HTTP {resp.status_code}: "
                f"{resp.json().get('detail', resp.text)}{colorama.Style.RESET_ALL}",
                file=sys.stderr,
            )
            return 1

        result = resp.json()
        result["seq"] = 1
        result["inference_ms"] = latency

        history: deque[dict] = deque(maxlen=HISTORY_WINDOW)
        history.append(result)

        print("\033c", end="")
        print_header(0, server_url, "HTTP 文件分析")
        render_display(result, history, 1, 1, False, 0.0, 0.0)
        return 0
    except Exception as e:
        print(
            f"{colorama.Fore.RED}[错误] {e}{colorama.Style.RESET_ALL}",
            file=sys.stderr,
        )
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="语音情感分类 - CLI客户端 (WebSocket 流式版)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # WebSocket实时流式识别（每500ms发一帧，服务端滑窗5秒推理）
  python client.py --server http://localhost:8000

  # 识别单个WAV文件（传统HTTP模式）
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
        "--history",
        type=int,
        default=HISTORY_WINDOW,
        help=f"历史曲线显示的窗口数 (默认: {HISTORY_WINDOW})",
    )

    args = parser.parse_args()
    colorama.init()

    if args.list_devices:
        list_devices()
        return

    if args.file:
        code = run_http_client(args.server, args.file)
        sys.exit(code)
    else:
        try:
            asyncio.run(
                run_stream_client(
                    args.server,
                    args.device,
                    history_size=args.history,
                )
            )
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
