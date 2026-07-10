import base64
import io
import json
import os

import httpx
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI()

# Allow all origins so the grader (whatever its origin is) is never blocked by CORS.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

AIPIPE_BASE = "https://aipipe.org/openai/v1"
AIPIPE_TOKEN = os.environ.get("AIPIPE_TOKEN", "")  # set this in Vercel env vars, never hardcode

EXTRACTION_SYSTEM_PROMPT = """You are a data extraction engine.
You will receive a transcript (in Korean, may include English/numbers) of someone describing
a small tabular dataset out loud: column names, data types, and row values.

Return ONLY a JSON object (no markdown, no prose) with this exact shape:
{
  "columns": ["col1", "col2", ...],
  "rows": [
    {"col1": value, "col2": value, ...},
    ...
  ]
}

Rules:
- Infer numeric vs categorical/string columns from context.
- Numeric values must be JSON numbers (int or float), not strings.
- If a value is ambiguous, make the single most reasonable interpretation.
- Do not invent extra columns or rows beyond what is described.
"""


async def transcribe_audio(audio_bytes: bytes) -> str:
    """Send audio to AI Pipe's Whisper (OpenAI-compatible) endpoint."""
    async with httpx.AsyncClient(timeout=8.0) as client:
        files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
        data = {"model": "whisper-1"}
        headers = {"Authorization": f"Bearer {AIPIPE_TOKEN}"}
        resp = await client.post(
            f"{AIPIPE_BASE}/audio/transcriptions",
            headers=headers,
            data=data,
            files=files,
        )
        resp.raise_for_status()
        return resp.json()["text"]


async def extract_table(transcript: str) -> dict:
    """Ask GPT (via AI Pipe) to turn the transcript into structured columns/rows."""
    async with httpx.AsyncClient(timeout=8.0) as client:
        headers = {
            "Authorization": f"Bearer {AIPIPE_TOKEN}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": transcript},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0,
        }
        resp = await client.post(
            f"{AIPIPE_BASE}/chat/completions",
            headers=headers,
            json=payload,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return json.loads(content)


def compute_stats(columns: list, rows: list) -> dict:
    df = pd.DataFrame(rows, columns=columns)

    numeric_df = df.select_dtypes(include="number")
    categorical_df = df.select_dtypes(exclude="number")

    def safe_dict(series_result):
        return {k: (None if pd.isna(v) else v) for k, v in series_result.items()}

    mean = safe_dict(numeric_df.mean(numeric_only=True))
    std = safe_dict(numeric_df.std(numeric_only=True))
    variance = safe_dict(numeric_df.var(numeric_only=True))
    minimum = safe_dict(numeric_df.min(numeric_only=True))
    maximum = safe_dict(numeric_df.max(numeric_only=True))
    median = safe_dict(numeric_df.median(numeric_only=True))

    mode = {}
    for col in numeric_df.columns:
        m = numeric_df[col].mode()
        mode[col] = m.iloc[0] if not m.empty else None

    value_range = {col: maximum.get(col, 0) - minimum.get(col, 0) for col in numeric_df.columns}
    range_ = value_range  # "range" and "value_range" both map to max-min per spec

    allowed_values = {
        col: sorted(df[col].dropna().unique().tolist()) for col in categorical_df.columns
    }

    correlation = []
    if len(numeric_df.columns) > 1:
        corr_matrix = numeric_df.corr(numeric_only=True).fillna(0)
        correlation = corr_matrix.values.tolist()

    return {
        "rows": len(df),
        "columns": list(df.columns),
        "mean": mean,
        "std": std,
        "variance": variance,
        "min": minimum,
        "max": maximum,
        "median": median,
        "mode": mode,
        "range": range_,
        "allowed_values": allowed_values,
        "value_range": value_range,
        "correlation": correlation,
    }


@app.post("/")
async def handle_audio(request: Request):
    body = await request.json()
    audio_id = body.get("audio_id", "")
    audio_b64 = body.get("audio_base64", "")

    audio_bytes = base64.b64decode(audio_b64)

    transcript = await transcribe_audio(audio_bytes)
    table = await extract_table(transcript)

    result = compute_stats(table["columns"], table["rows"])
    return JSONResponse(content=result)


@app.get("/")
async def health():
    return {"status": "ok"}
