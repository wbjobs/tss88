import io
import wave
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from server import (
    ALL_EMOTION_LABELS,
    AudioConfig,
    EmotionModel,
    EMOTION_LABELS,
    SILENT_LABEL,
    detect_silence_from_wav_bytes,
    preprocess_audio,
)


MODEL_PATH = Path("models/emotion_classifier.onnx")
MAX_AUDIO_SIZE = int(AudioConfig.MAX_DURATION * 48000 * 2 * 1 + 4096)
ALLOWED_EXTENSIONS = {".wav"}


class EmotionResponse(BaseModel):
    emotions: dict[str, float] = Field(
        description="所有情感类别（含silent）的概率分布，总和为1"
    )
    dominant: str = Field(description="占主导的情感类别")
    confidence: float = Field(description="主导情感的置信度，范围 0~1")
    is_silent: bool = Field(description="是否被判定为静音段")
    vad_speech_ratio: float = Field(description="VAD语音帧占比（0~1）")
    vad_avg_rms: float = Field(description="音频平均RMS能量")


_model: EmotionModel | None = None


def get_model() -> EmotionModel:
    global _model
    if _model is None:
        raise RuntimeError("Model is not loaded")
    return _model


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    if not MODEL_PATH.exists():
        print(
            f"[WARNING] Model file not found at {MODEL_PATH}. "
            "Please place a trained ONNX model there, or run "
            "`python scripts/generate_dummy_model.py` to create a test model."
        )
    else:
        print(f"[INFO] Loading ONNX model from: {MODEL_PATH}")
        _model = EmotionModel(MODEL_PATH)
        providers = _model.session.get_providers()
        print(f"[INFO] Model loaded successfully. Using providers: {providers}")
    yield
    print("[INFO] Shutting down...")


app = FastAPI(
    title="Speech Emotion Classification API",
    description="AI service for speech emotion classification using ONNX model. "
    "Input: WAV audio (up to 5s). Output: probability distribution over "
    "5 emotion classes (happy, sad, angry, neutral, silent). "
    "Built-in VAD to detect silence segments.",
    version="1.1.0",
    lifespan=lifespan,
)


def _validate_wav(data: bytes) -> tuple[int, int, int, float]:
    try:
        with wave.open(io.BytesIO(data), "rb") as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            frame_rate = wf.getframerate()
            n_frames = wf.getnframes()
            duration = n_frames / frame_rate if frame_rate > 0 else 0.0

            if sample_width != 2:
                raise ValueError(
                    f"Only 16-bit PCM WAV is supported (got {sample_width * 8}-bit)"
                )
            if channels not in (1, 2):
                raise ValueError(f"Only mono or stereo WAV is supported (got {channels} channels)")
            if duration <= 0:
                raise ValueError("Audio duration is zero")
            if duration > AudioConfig.MAX_DURATION + 0.5:
                raise ValueError(
                    f"Audio too long: {duration:.2f}s. Max allowed: {AudioConfig.MAX_DURATION}s"
                )
            return channels, sample_width, frame_rate, duration
    except wave.Error as e:
        raise ValueError(f"Invalid WAV file: {e}")


def _build_silent_response(speech_ratio: float, avg_rms: float) -> EmotionResponse:
    emotions = {k: 0.0 for k in EMOTION_LABELS}
    emotions[SILENT_LABEL] = 1.0
    return EmotionResponse(
        emotions=emotions,
        dominant=SILENT_LABEL,
        confidence=1.0,
        is_silent=True,
        vad_speech_ratio=speech_ratio,
        vad_avg_rms=avg_rms,
    )


@app.get("/")
async def root():
    return {
        "service": "Speech Emotion Classification API",
        "version": "1.1.0",
        "emotions": ALL_EMOTION_LABELS,
        "vad_thresholds": {
            "rms": AudioConfig.VAD_RMS_THRESHOLD,
            "speech_ratio": AudioConfig.VAD_SPEECH_RATIO_THRESHOLD,
        },
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST, multipart/form-data, field: 'file')",
        },
    }


@app.get("/health")
async def health():
    model_loaded = _model is not None
    status = "healthy" if model_loaded else "degraded"
    return {
        "status": status,
        "model_loaded": model_loaded,
        "emotions": ALL_EMOTION_LABELS,
    }


@app.post("/predict", response_model=EmotionResponse)
async def predict(file: UploadFile = File(...)):
    filename = file.filename or ""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {ext or 'unknown'}. Allowed: {ALLOWED_EXTENSIONS}",
        )

    data = await file.read()
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")
    if len(data) > MAX_AUDIO_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large: {len(data)} bytes. Max allowed: {MAX_AUDIO_SIZE} bytes",
        )

    try:
        channels, sample_width, sample_rate, duration = _validate_wav(data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        is_silent, speech_ratio, avg_rms = detect_silence_from_wav_bytes(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"VAD analysis failed: {e}")

    if is_silent:
        return JSONResponse(
            content=_build_silent_response(speech_ratio, avg_rms).model_dump()
        )

    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please place the ONNX model at models/emotion_classifier.onnx",
        )

    try:
        mel_spec = preprocess_audio(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio preprocessing failed: {e}")

    try:
        emotion_probs = _model.predict(mel_spec)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

    emotion_probs[SILENT_LABEL] = 0.0

    dominant = max(emotion_probs, key=emotion_probs.get)
    confidence = emotion_probs[dominant]

    return JSONResponse(
        content=EmotionResponse(
            emotions=emotion_probs,
            dominant=dominant,
            confidence=confidence,
            is_silent=False,
            vad_speech_ratio=speech_ratio,
            vad_avg_rms=avg_rms,
        ).model_dump()
    )
