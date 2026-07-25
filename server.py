from __future__ import annotations

import json
import os
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import librosa
import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from openai import OpenAI
from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text, create_engine, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

APP_NAME = "Church Audio AI Assistant"
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "75"))
ANALYSIS_SECONDS = min(float(os.getenv("ANALYSIS_SECONDS", "15")), 15.0)
ANALYSIS_SAMPLE_RATE = min(int(os.getenv("ANALYSIS_SAMPLE_RATE", "11025")), 11025)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def database_url() -> str:
    url = os.getenv("DATABASE_URL", "sqlite:///./church_audio.db")
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+psycopg://", 1)
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


connect_args = {"check_same_thread": False} if database_url().startswith("sqlite") else {}
engine = create_engine(database_url(), pool_pre_ping=True, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class AudioUpload(Base):
    __tablename__ = "audio_uploads"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    content_type: Mapped[str] = mapped_column(String(120), nullable=False)
    storage_path: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(40), default="uploaded", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    report: Mapped[SoundReport | None] = relationship(
        back_populates="upload",
        cascade="all, delete-orphan",
        uselist=False,
    )


class SoundReport(Base):
    __tablename__ = "sound_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    upload_id: Mapped[int] = mapped_column(ForeignKey("audio_uploads.id"), unique=True, nullable=False)
    source: Mapped[str] = mapped_column(String(40), nullable=False)
    features: Mapped[dict] = mapped_column(JSON, nullable=False)
    report_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    upload: Mapped[AudioUpload] = relationship(back_populates="report")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def clean_float(value: float, digits: int = 3) -> float:
    if np.isnan(value) or np.isinf(value):
        return 0.0
    return round(float(value), digits)


def extract_features(path: str | Path) -> dict:
    y, sr = librosa.load(path, sr=ANALYSIS_SAMPLE_RATE, mono=True, duration=ANALYSIS_SECONDS)
    if y.size == 0:
        raise ValueError("No audio samples could be decoded from this file.")

    duration = librosa.get_duration(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=1.0)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    peak_db = 20 * np.log10(max(peak, 1e-9))
    crest_factor = peak / max(float(np.mean(rms)), 1e-9)

    spectrum = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(y.size, d=1 / sr)
    low_energy = float(spectrum[(freqs >= 20) & (freqs < 250)].mean()) if spectrum.size else 0.0
    mid_energy = float(spectrum[(freqs >= 250) & (freqs < 4000)].mean()) if spectrum.size else 0.0
    high_energy = float(spectrum[(freqs >= 4000) & (freqs < 5500)].mean()) if spectrum.size else 0.0

    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    clipping_ratio = float(np.mean(np.abs(y) > 0.98)) if y.size else 0.0
    silence_ratio = float(np.mean(rms_db < -55)) if rms_db.size else 0.0

    return {
        "analysis_window_seconds": clean_float(ANALYSIS_SECONDS, 2),
        "duration_seconds": clean_float(duration, 2),
        "sample_rate_hz": int(sr),
        "loudness": {
            "rms_db_mean": clean_float(np.mean(rms_db)),
            "rms_db_min": clean_float(np.min(rms_db)),
            "rms_db_max": clean_float(np.max(rms_db)),
            "peak_db": clean_float(peak_db),
            "crest_factor": clean_float(crest_factor),
            "clipping_ratio": clean_float(clipping_ratio, 5),
            "silence_ratio": clean_float(silence_ratio, 4),
        },
        "spectrum": {
            "spectral_centroid_hz": clean_float(np.mean(spectral_centroid)),
            "low_energy": clean_float(low_energy),
            "mid_energy": clean_float(mid_energy),
            "high_energy": clean_float(high_energy),
            "low_to_mid_ratio": clean_float(low_energy / max(mid_energy, 1e-9)),
            "high_to_mid_ratio": clean_float(high_energy / max(mid_energy, 1e-9)),
        },
    }


def rule_report(features: dict) -> dict:
    loudness = features["loudness"]
    spectrum = features["spectrum"]
    diagnostics = []
    recommendations = []
    priority = "low"
    grade = "A"

    if loudness["clipping_ratio"] > 0.002 or loudness["peak_db"] > -0.5:
        priority = "high"
        grade = "D"
        diagnostics.append({
            "area": "loudness",
            "finding": "The recording is very close to clipping or already clipping.",
            "evidence": f"Peak level is {loudness['peak_db']} dBFS with clipping ratio {loudness['clipping_ratio']}.",
            "severity": "high",
        })
        recommendations.append({
            "title": "Create more capture headroom.",
            "reason": "Clipping cannot be repaired cleanly after capture.",
            "priority": "high",
        })

    if loudness["rms_db_mean"] < -35:
        priority = "medium" if priority == "low" else priority
        grade = "C" if grade == "A" else grade
        diagnostics.append({
            "area": "loudness",
            "finding": "Average program level is low.",
            "evidence": f"Mean RMS is {loudness['rms_db_mean']} dBFS.",
            "severity": "medium",
        })
        recommendations.append({
            "title": "Raise recording gain while preserving peak headroom.",
            "reason": "Low capture level makes speech harder to understand.",
            "priority": "medium",
        })

    if spectrum["low_to_mid_ratio"] > 1.35:
        priority = "medium" if priority == "low" else priority
        grade = "C" if grade == "A" else grade
        diagnostics.append({
            "area": "frequency_balance",
            "finding": "Low-frequency energy may be masking speech intelligibility.",
            "evidence": f"Low-to-mid energy ratio is {spectrum['low_to_mid_ratio']}.",
            "severity": "medium",
        })
        recommendations.append({
            "title": "Check low-frequency buildup from bass, kick, room resonance, and lectern microphones.",
            "reason": "Reducing unnecessary low end improves clarity.",
            "priority": "medium",
        })

    if spectrum["high_to_mid_ratio"] < 0.22:
        diagnostics.append({
            "area": "clarity",
            "finding": "The recording may sound dull or lack presence.",
            "evidence": f"High-to-mid energy ratio is {spectrum['high_to_mid_ratio']}.",
            "severity": "low",
        })
        recommendations.append({
            "title": "Check microphone placement before adding global brightness.",
            "reason": "Presence helps lyrics and sermon detail translate on phones.",
            "priority": "low",
        })

    if not diagnostics:
        diagnostics.append({
            "area": "clarity",
            "finding": "No major technical fault was detected from extracted features.",
            "evidence": "Loudness, clipping, and broad frequency balance are within MVP thresholds.",
            "severity": "low",
        })
        recommendations.append({
            "title": "Use this report as a baseline.",
            "reason": "Comparing future services helps reveal setup or routing changes.",
            "priority": "low",
        })

    return {
        "overall_grade": grade,
        "priority": priority,
        "summary": "Rule-based diagnosis generated from audio features. Add OPENAI_API_KEY for AI-authored analysis.",
        "diagnostics": diagnostics,
        "recommendations": recommendations,
        "json_version": "1.0",
    }


def openai_report(features: dict) -> tuple[dict, str]:
    if not OPENAI_API_KEY:
        return rule_report(features), "rules"

    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt = {
        "task": "Analyze church service audio features and return strict JSON only.",
        "schema": {
            "overall_grade": "A|B|C|D",
            "priority": "low|medium|high",
            "summary": "short diagnosis",
            "diagnostics": [{"area": "string", "finding": "string", "evidence": "string", "severity": "low|medium|high"}],
            "recommendations": [{"title": "string", "reason": "string", "priority": "low|medium|high"}],
            "json_version": "1.0",
        },
        "features": features,
        "restriction": "Do not implement or suggest mixer control automation. Focus on diagnosis and recommendations.",
    }
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are a senior live sound engineer for church services. Return strict JSON only."},
            {"role": "user", "content": json.dumps(prompt)},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    return json.loads(response.choices[0].message.content or "{}"), "openai"


app = FastAPI(title=APP_NAME)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    Base.metadata.create_all(bind=engine)


@app.get("/")
def home() -> FileResponse:
    return FileResponse("index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/debug")
def debug() -> dict[str, float | int | str]:
    return {
        "status": "ok",
        "max_upload_mb": MAX_UPLOAD_MB,
        "analysis_seconds": ANALYSIS_SECONDS,
        "analysis_sample_rate": ANALYSIS_SAMPLE_RATE,
        "openai_enabled": "yes" if OPENAI_API_KEY else "no",
    }


@app.post("/api/uploads")
def upload_audio(file: UploadFile = File(...), db: Session = Depends(get_db)) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported audio type.")

    total = 0
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    tmp_path = None
    upload = AudioUpload(
        filename=file.filename or f"{uuid4().hex}{suffix}",
        content_type=file.content_type or "application/octet-stream",
        storage_path="discarded_after_analysis",
        status="processing",
    )
    db.add(upload)
    db.commit()
    db.refresh(upload)

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            while chunk := file.file.read(1024 * 1024):
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(status_code=413, detail="Audio file is too large.")
                tmp.write(chunk)

        features = extract_features(tmp_path)
        report_json, source = openai_report(features)
        report = SoundReport(upload_id=upload.id, features=features, report_json=report_json, source=source)
        upload.status = "complete"
        db.add(report)
        db.commit()
        db.refresh(report)
        return {
            "upload_id": upload.id,
            "report_id": report.id,
            "status": upload.status,
            "source": report.source,
            "report": report.report_json,
        }
    except HTTPException:
        upload.status = "failed"
        db.commit()
        raise
    except Exception as exc:
        print("Audio analysis failed")
        traceback.print_exc()
        upload.status = "failed"
        db.commit()
        raise HTTPException(status_code=500, detail=f"Audio analysis failed: {exc}") from exc
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


@app.get("/api/reports")
def list_reports(db: Session = Depends(get_db)) -> list[dict]:
    rows = db.execute(
        select(SoundReport, AudioUpload)
        .join(AudioUpload, SoundReport.upload_id == AudioUpload.id)
        .order_by(SoundReport.created_at.desc())
    ).all()
    return [
        {
            "id": report.id,
            "upload_id": upload.id,
            "filename": upload.filename,
            "source": report.source,
            "created_at": report.created_at,
            "overall_grade": report.report_json.get("overall_grade", "C"),
            "priority": report.report_json.get("priority", "medium"),
        }
        for report, upload in rows
    ]


@app.get("/api/reports/{report_id}")
def get_report(report_id: int, db: Session = Depends(get_db)) -> dict:
    row = db.execute(
        select(SoundReport, AudioUpload)
        .join(AudioUpload, SoundReport.upload_id == AudioUpload.id)
        .where(SoundReport.id == report_id)
    ).first()
    if not row:
        raise HTTPException(status_code=404, detail="Report not found.")
    report, upload = row
    return {
        "id": report.id,
        "upload_id": upload.id,
        "filename": upload.filename,
        "source": report.source,
        "created_at": report.created_at,
        "features": report.features,
        "report": report.report_json,
    }
