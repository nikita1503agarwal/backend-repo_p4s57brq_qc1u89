import os
import io
import uuid
import math
import struct
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import wave

app = FastAPI(title="Enterprise Speech Demo API", version="1.0.1")

# CORS (open for demo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure output dir exists and mount as static for serving generated audio
GENERATED_DIR = os.path.join(os.getcwd(), "generated")
os.makedirs(GENERATED_DIR, exist_ok=True)
app.mount("/generated", StaticFiles(directory=GENERATED_DIR), name="generated")


@app.get("/")
def read_root():
    return {"message": "Speech API is running", "endpoints": ["/api/tts", "/api/asr", "/api/asr/batch", "/api/voices", "/test"]}


@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}


@app.get("/test")
def test_database():
    """Simple health endpoint (database optional in this app)."""
    response = {
        "backend": "✅ Running",
        "database": "ℹ️ Not used in this demo",
        "database_url": "❌ Not Set",
        "database_name": "❌ Not Set",
        "connection_status": "N/A",
        "collections": [],
    }
    try:
        import os as _os
        response["database_url"] = "✅ Set" if _os.getenv("DATABASE_URL") else "❌ Not Set"
        response["database_name"] = "✅ Set" if _os.getenv("DATABASE_NAME") else "❌ Not Set"
    except Exception:
        pass
    return response


# --------- TTS (Tone-based demo synthesis without external deps) ---------
class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    rate: Optional[int] = 180


def synth_tone_speech(text: str, sample_rate: int = 22050) -> bytes:
    """
    Very lightweight demo TTS: generates a tone sequence that encodes characters
    as different frequencies. Pure Python (no numpy) for maximum portability.
    """
    duration_per_char = 0.12  # seconds
    silence_gap = 0.05

    # Map characters to frequencies (basic A-Z, 0-9, space)
    base_freq = 440.0  # A4
    char_map = {chr(ord('A') + i): base_freq + i * 20 for i in range(26)}
    num_map = {str(i): 300 + i * 25 for i in range(10)}
    char_map.update(num_map)
    char_map[' '] = 0  # silence for spaces
    char_map['.'] = 660
    char_map[','] = 520

    samples = []  # 16-bit signed ints
    n_char = int(sample_rate * duration_per_char)
    n_gap = int(sample_rate * silence_gap)

    for ch in text.upper():
        freq = char_map.get(ch, 0)
        if freq > 0:
            for n in range(n_char):
                # 0.5 amplitude to avoid clipping when concatenating
                value = int(0.5 * 32767 * math.sin(2.0 * math.pi * freq * (n / sample_rate)))
                samples.append(value)
        else:
            # silence for unknowns/spaces
            samples.extend([0] * n_char)
        # inter-character gap
        samples.extend([0] * n_gap)

    if not samples:
        samples = [0] * int(sample_rate * 0.2)

    # Pack to WAV bytes
    with io.BytesIO() as buf:
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(b''.join(struct.pack('<h', s) for s in samples))
        return buf.getvalue()


@app.post("/api/tts")
def tts(req: TTSRequest):
    if not req.text or not req.text.strip():
        return JSONResponse(status_code=400, content={"error": "text is required"})

    wav_bytes = synth_tone_speech(req.text.strip())
    file_id = str(uuid.uuid4())
    filename = f"tts_{file_id}.wav"
    file_path = os.path.join(GENERATED_DIR, filename)
    with open(file_path, "wb") as f:
        f.write(wav_bytes)

    return {
        "audio_url": f"/generated/{filename}",
        "format": "wav",
        "bytes": len(wav_bytes),
        "note": "Demo tone synthesis (not natural speech)",
    }


# --------- ASR (Analysis-only demo, pure Python) ---------

def _rms_ints(int_iterable) -> float:
    # Compute RMS of iterable of ints without external deps
    # RMS = sqrt(mean(x^2))
    total = 0
    count = 0
    for v in int_iterable:
        total += v * v
        count += 1
    if count == 0:
        return 0.0
    return (total / count) ** 0.5


def analyze_audio(file_bytes: bytes) -> dict:
    """
    Demo analysis: extracts basic properties (duration) and RMS level.
    Works with standard 16-bit PCM WAV best; provides limited info otherwise.
    """
    try:
        with io.BytesIO(file_bytes) as buf:
            with wave.open(buf, 'rb') as wf:
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                framerate = wf.getframerate()
                n_frames = wf.getnframes()
                duration = n_frames / float(framerate) if framerate else 0.0

                # Only handle 16-bit and 32-bit PCM for RMS
                frames = wf.readframes(n_frames)
                ints = []
                if sampwidth == 2:
                    # little-endian 16-bit signed
                    for i in range(0, len(frames), 2):
                        (val,) = struct.unpack_from('<h', frames, i)
                        ints.append(val)
                elif sampwidth == 4:
                    # 32-bit signed (downscale to 16-bit range for RMS normalization)
                    for i in range(0, len(frames), 4):
                        (val32,) = struct.unpack_from('<i', frames, i)
                        val16 = max(-32768, min(32767, val32 // 65536))
                        ints.append(val16)
                else:
                    ints = []

                # If stereo, downmix to mono by averaging pairs
                if n_channels > 1 and ints:
                    mono = []
                    for i in range(0, len(ints), n_channels):
                        chunk = ints[i:i+n_channels]
                        if len(chunk) == n_channels:
                            mono.append(sum(chunk) // n_channels)
                    ints = mono

                rms = _rms_ints(ints) if ints else 0.0
                return {
                    "duration_seconds": round(duration, 3),
                    "sample_rate": framerate,
                    "channels": n_channels,
                    "rms_level": round(float(rms), 2),
                }
    except wave.Error:
        return {"note": "Non-WAV or unsupported WAV. For best results, upload 16-bit PCM WAV."}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/asr")
async def asr(file: UploadFile = File(...)):
    content = await file.read()
    info = analyze_audio(content)
    return {
        "filename": file.filename,
        "info": info,
        "transcription": "Demo environment: no speech model available.",
    }


@app.post("/api/asr/batch")
async def asr_batch(files: List[UploadFile] = File(...)):
    results = []
    for f in files:
        content = await f.read()
        info = analyze_audio(content)
        results.append({
            "filename": f.filename,
            "info": info,
            "transcription": "Demo environment: no speech model available.",
        })
    return {"results": results}


@app.get("/api/voices")
def voices():
    return {
        "voices": [
            {"id": "demo_male", "name": "Demo Male", "language": "en-US"},
            {"id": "demo_female", "name": "Demo Female", "language": "en-US"},
        ]
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
