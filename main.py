import io
import wave
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from server import AudioConfig, EmotionModel, EMOTION_LABELS, preprocess_audio


MODEL_PATH = Path("models/emotion_classifier.onnx")
MAX_AUDIO_SIZE = 5 * 16000 * 2 * 1 + 1024
ALLOWED_EXTENSIONS = {".wav"}


class EmotionResponse(BaseModel):
    emotions: dict[str, float]
    dominant: str
    confidence: float


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
        print(f"[INFO] Model loaded successfully. Using providers: {_model.session.get_providers()}")
    yield
    print("[INFO] Shutting down...")


app = FastAPI(
    title="Speech Emotion Classification API",
    description="AI service for speech emotion classification using ONNX model. "
    "Input: WAV audio (up to 5s). Output: probability distribution over "
    "4 emotion classes (happy, sad, angry, neutral).",
    version="1.0.0",
    lifespan=lifespan,
)


def _validate_wav(data: bytes) -> tuple[int, float]:
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
            if duration > AudioConfig.MAX_DURATION:
                raise ValueError(
                    f"Audio too long: {duration:.2f}s. Max allowed: {AudioConfig.MAX_DURATION}s"
                )
            return frame_rate, duration
    except wave.Error as e:
        raise ValueError(f"Invalid WAV file: {e}")


@app.get("/")
async def root():
    return {
        "service": "Speech Emotion Classification API",
        "version": "1.0.0",
        "emotions": EMOTION_LABELS,
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
        "emotions": EMOTION_LABELS,
    }


@app.post("/predict", response_model=EmotionResponse)
async def predict(file: UploadFile = File(...)):
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please place the ONNX model at models/emotion_classifier.onnx",
        )

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
        sample_rate, duration = _validate_wav(data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        mel_spec = preprocess_audio(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio preprocessing failed: {e}")

    try:
        emotion_probs = _model.predict(mel_spec)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

    dominant = max(emotion_probs, key=emotion_probs.get)
    confidence = emotion_probs[dominant]

    return JSONResponse(
        content=EmotionResponse(
            emotions=emotion_probs,
            dominant=dominant,
            confidence=confidence,
        ).model_dump()
    )
