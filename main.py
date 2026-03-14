
import json
import os
import tempfile
from math import gcd
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from scipy.signal import resample_poly
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification




APP_TITLE = "Emotion Detection API"
DEFAULT_HF_MODEL = "superb/wav2vec2-base-superb-er"
REPO_CHECKPOINT_PATH = Path("Emotion-detection-/experiment/checkpoints/best_model.pt")
REPO_LABELS_PATH = Path("Emotion-detection-/experiment/labels.json")
SAMPLE_RATE = 16000


class EmotionEngine:
    """Loads an emotion model and returns emotion labels from audio files."""

    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mode = "transformers-wav2vec2"
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(DEFAULT_HF_MODEL)
        self.model = AutoModelForAudioClassification.from_pretrained(
            DEFAULT_HF_MODEL,
            use_safetensors=False,
        )
        self.model.to(self.device)
        self.model.eval()

        # Optional: if the user adds repo checkpoints, switch to local mode.
        self.local_model = None
        self.local_labels: list[str] | None = None
        if REPO_CHECKPOINT_PATH.exists() and REPO_LABELS_PATH.exists():
            try:
                with REPO_LABELS_PATH.open("r", encoding="utf-8") as f:
                    labels = json.load(f)
                if isinstance(labels, list) and labels:
                    model = torch.jit.load(str(REPO_CHECKPOINT_PATH), map_location=self.device)
                    model.eval()
                    self.local_model = model
                    self.local_labels = labels
                    self.mode = "repo-local"
            except Exception:
                # Keep transformer mode if optional local artifacts are not usable.
                self.local_model = None
                self.local_labels = None

    def _load_waveform(self, audio_path: str) -> torch.Tensor:
        data, sr = sf.read(audio_path, always_2d=True)
        if data.size == 0:
            raise ValueError("Audio decode failed or empty audio data.")

        # Convert multi-channel audio to mono.
        mono = data.mean(axis=1).astype(np.float32)

        if sr != SAMPLE_RATE:
            factor = gcd(sr, SAMPLE_RATE)
            up = SAMPLE_RATE // factor
            down = sr // factor
            mono = resample_poly(mono, up, down).astype(np.float32)

        return torch.from_numpy(mono).unsqueeze(0)

    def predict(self, audio_path: str) -> dict[str, Any]:
        waveform = self._load_waveform(audio_path)

        if (
            self.mode == "repo-local"
            and self.local_model is not None
            and self.local_labels is not None
        ):
            with torch.no_grad():
                logits = self.local_model(waveform.to(self.device))
                probs = torch.softmax(logits, dim=-1)
                best_idx = int(torch.argmax(probs, dim=-1).item())
                best_score = float(probs[0, best_idx].item())

            return {
                "emotion": self.local_labels[best_idx],
                "score": round(best_score, 4),
                "model_mode": self.mode,
            }

        with torch.no_grad():
            mono_waveform = waveform.squeeze(0).cpu().numpy()
            inputs = self.feature_extractor(
                mono_waveform,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            best_idx = int(torch.argmax(probs, dim=-1).item())
            confidence = float(probs[0, best_idx].item())

        label = self.model.config.id2label.get(best_idx, str(best_idx))

        return {
            "emotion": label,
            "score": round(confidence, 4),
            "model_mode": self.mode,
            "raw_index": best_idx,
        }


app = FastAPI(title=APP_TITLE)
engine: EmotionEngine | None = None


@app.on_event("startup")
async def on_startup() -> None:
    global engine
    print("[SERVER] Booting Emotion API...", flush=True)
    print("[SERVER] Loading emotion model (first start can take time)...", flush=True)
    engine = EmotionEngine()
    print(f"[SERVER] Model loaded successfully (mode={engine.mode})", flush=True)
    print("[SERVER] Emotion API running on http://0.0.0.0:8000", flush=True)
    print("[SERVER] Waiting for audio on POST /analyze-audio", flush=True)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "mode": engine.mode if engine is not None else "loading"}


@app.post("/analyze-audio")
async def analyze_audio(file: UploadFile = File(...)) -> dict[str, Any]:
    if engine is None:
        raise HTTPException(status_code=503, detail="Modele en cours de chargement. Reessaye dans quelques secondes.")

    ext = Path(file.filename or "audio.wav").suffix or ".wav"
    temp_path: str | None = None

    try:
        print(f"[SERVER] Incoming audio request: file={file.filename}", flush=True)
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            temp_path = temp_file.name
            content = await file.read()
            if not content:
                raise HTTPException(status_code=400, detail="Fichier audio vide.")
            temp_file.write(content)

        result = engine.predict(temp_path)

        # Required by your request: print the answer in the computer terminal.
        print(
            f"[EMOTION] file={file.filename} -> emotion={result['emotion']} "
            f"score={result['score']} mode={result['model_mode']}",
            flush=True,
        )
        print("[SERVER] Waiting for next audio request...", flush=True)

        return {
            "file": file.filename,
            **result,
        }
    except HTTPException:
        raise
    except Exception as exc:
        print(f"[SERVER] Inference error: {exc}", flush=True)
        raise HTTPException(status_code=500, detail=f"Erreur inférence audio: {exc}") from exc
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)