#!/usr/bin/env python3
import argparse
import asyncio
import csv
import json
import os
import re
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict
from pydantic import BaseModel, Field

try:
    import google.genai as genai
except ImportError:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
else:
    try:
        from google.genai.types import HarmCategory, HarmBlockThreshold
    except ImportError:
        pass

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

from yt_dlp import DownloadError, YoutubeDL

class TranscriptUnavailable(Exception):
    pass

# --- Data Schemas for Gemini ---

class Question(BaseModel):
    type: str = Field(description="Must be one of: MCQ, MSQ, FILL_BLANK, FLASHCARD")
    question: str
    options: List[str] = Field(description="List of strings, empty for FILL_BLANK/FLASHCARD")
    correct_answer: str
    explanation: str

class QuizList(BaseModel):
    questions: List[Question]

@dataclass
class TranscriptEntry:
    minute_index: int
    text: str

# --- Processing Logic ---

def extract_video_id(url: str) -> str:
    patterns = [
        r"(?:v=|youtu\.be/|youtube\.com/(?:embed/|shorts/))([A-Za-z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError("Could not parse YouTube video ID from URL")


def fetch_transcript(video_id: str) -> List[Dict]:
    url = f"https://www.youtube.com/watch?v={video_id}"
    with tempfile.TemporaryDirectory() as tmpdir:
        ydl_opts = {
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": ["en"],
            "subtitlesformat": "json3",
            "outtmpl": os.path.join(tmpdir, "%(id)s.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
        }
        try:
            with YoutubeDL(ydl_opts) as ydl:
                ydl.extract_info(url, download=True)
        except DownloadError as exc:
            raise TranscriptUnavailable(f"Transcript download failed: {exc}") from exc

        transcript_files = list(Path(tmpdir).glob("*.json3"))
        if not transcript_files:
            raise TranscriptUnavailable("No English subtitles found via yt-dlp")

        entries = []
        for file_path in transcript_files:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            events = data.get("events") or []
            for event in events:
                start_ms = event.get("tStartMs")
                segs = event.get("segs") or []
                parts = [seg.get("utf8", "").strip() for seg in segs if seg.get("utf8")]
                text = " ".join(parts).strip()
                if text and start_ms is not None:
                    entries.append({"start": start_ms / 1000.0, "text": text})

        if not entries:
            raise TranscriptUnavailable("No transcript text could be parsed from subtitles")

        entries.sort(key=lambda item: item["start"])
        return entries


def build_transcript_chunks(entries: List[Dict], batch_size_minutes: int = 5) -> List[Dict]:
    minutes = defaultdict(list)
    for entry in entries:
        start = entry.get("start", 0)
        text = entry.get("text", "").strip()
        if not text:
            continue
        minute_index = int(start // 60)
        minutes[minute_index].append(text)

    if not minutes:
        return []

    chunked = []
    min_minute = min(minutes.keys())
    max_minute = max(minutes.keys())

    for bucket_start in range(min_minute, max_minute + 1, batch_size_minutes):
        texts = []
        bucket_end = bucket_start + batch_size_minutes
        for minute in range(bucket_start, bucket_end):
            if minute in minutes:
                texts.extend(minutes[minute])

        if texts:
            chunked.append({
                "start_min": bucket_start,
                "end_min": bucket_end,
                "text": "\n".join(texts),
            })

    return chunked


async def generate_questions_for_chunk(model, chunk_text: str, start_min: int, end_min: int, q_per_min: int) -> List[Dict]:
    """Generates questions for a 5-minute chunk using Gemini."""
    num_questions = q_per_min * max(1, end_min - start_min)

    prompt = f"""
    You are an expert tutor. Create {num_questions} quiz questions based on this lecture segment ({start_min}m to {end_min}m).
    Ensure a mix of MCQ, MSQ, FILL_BLANK, and FLASHCARD types.

    Transcript:
    {chunk_text}
    """

    try:
        result = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json",
                response_schema=QuizList,
            ),
        )
        data = json.loads(result.text)

        rows = []
        for q in data.get("questions", []):
            rows.append({
                "minute_start": start_min,
                "minute_end": end_min,
                "type": q.get("type", "MCQ"),
                "question": q.get("question", ""),
                "options": " | ".join(q.get("options", [])),
                "correct_answer": q.get("correct_answer", ""),
                "explanation": q.get("explanation", ""),
            })
        return rows
    except Exception as e:
        print(f"Error processing {start_min}-{end_min}: {e}", file=sys.stderr)
        return []

async def main():
    parser = argparse.ArgumentParser(description="Efficient YouTube-to-Quiz via Gemini")
    parser.add_argument("url")
    parser.add_argument("--api-key", default=os.environ.get("GOOGLE_API_KEY"))
    parser.add_argument("--qpm", type=int, default=3, help="Questions per minute")
    parser.add_argument("--output", default="quiz.csv")
    parser.add_argument("--batch-minutes", type=int, default=5, help="Minutes per chunk")
    args = parser.parse_args()

    if not args.api_key:
        print("Set GOOGLE_API_KEY environment variable."); sys.exit(1)

    genai.configure(api_key=args.api_key)
    model = genai.GenerativeModel("gemini-3.5-flash")

    print("Fetching transcript...")
    try:
        video_id = extract_video_id(args.url)
        entries = fetch_transcript(video_id)
    except TranscriptUnavailable as e:
        print(f"Failed to fetch transcript: {e}")
        sys.exit(1)
    except ValueError as e:
        print(str(e))
        sys.exit(1)
    except Exception as e:
        print(f"Failed to fetch transcript: {e}")
        sys.exit(1)

    chunks = build_transcript_chunks(entries, batch_size_minutes=args.batch_minutes)
    if not chunks:
        print("No transcript text found after parsing.")
        sys.exit(1)

    tasks = []
    for chunk in chunks:
        tasks.append(
            generate_questions_for_chunk(
                model,
                chunk["text"],
                chunk["start_min"],
                chunk["end_min"],
                args.qpm,
            )
        )

    print(f"Processing {len(tasks)} batches concurrently...")
    results = await asyncio.gather(*tasks)

    all_rows = [item for sublist in results for item in sublist]
    if not all_rows:
        print("No questions were generated.")
        sys.exit(1)

    with open(args.output, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["minute_start", "minute_end", "type", "question", "options", "correct_answer", "explanation"],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Done! Generated {len(all_rows)} questions and wrote to {args.output}.")

if __name__ == "__main__":
    asyncio.run(main())