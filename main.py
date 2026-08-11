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
from openrouter import OpenRouter
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

from yt_dlp import DownloadError, YoutubeDL

class TranscriptUnavailable(Exception):
    pass

# --- Data Schemas for Gemini ---

class Question(BaseModel):
    question_type: str = Field(description="Must be one of: mcq, multi_select, fill_blank, flashcard")
    question: str
    options: List[str] = Field(description="List of option strings, empty list for fill_blank/flashcard")
    correct_answers: List[str] = Field(description="List of correct answer strings. For mcq/fill_blank/flashcard use a single-element list. For multi_select use multiple elements.")
    explanation: str
    tags: List[str] = Field(description="List of topic/category tags for this question, e.g. ['python', 'data types']")

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


# Map from Gemini schema type names to desired CSV type names
TYPE_MAP = {
    "mcq": "mcq",
    "multi_select": "multi_select",
    "fill_blank": "fill_blank",
    "flashcard": "flashcard",
}


def generate_questions_for_chunk(client, chunk_text: str, start_min: int, end_min: int, q_per_min: int) -> List[Dict]:
    """Generates questions for a 5-minute chunk using Groq."""
    num_questions = q_per_min * max(1, end_min - start_min)
    print(f"[{start_min}m-{end_min}m] Requesting {num_questions} questions from OpenRouter...")

    prompt = f"""
    You are an expert tutor. Create {num_questions} quiz questions based on this lecture segment ({start_min}m to {end_min}m).
    Ensure a mix of question types: mcq, multi_select, fill_blank, and flashcard.

    Rules:
    - question_type must be one of: mcq, multi_select, fill_blank, flashcard
    - For mcq: provide 3-5 options and exactly one correct answer
    - For multi_select: provide 3-5 options and multiple correct answers
    - For fill_blank: leave options as an empty list, provide the blank answer in correct_answers
    - For flashcard: leave options as an empty list, provide the answer in correct_answers
    - tags: provide 2-3 relevant topic/category tags per question

    OUTPUT FORMAT: You MUST return a valid JSON object with a single key "questions" containing a list of objects matching the fields described above.

    Transcript:
    {chunk_text}
    """

    try:
        response = client.chat.send(
            model="nvidia/nemotron-3-super-120b-a12b:free",
            messages=[
              {
                "role": "user",
                "content": prompt
              }
            ]
        )
        result_text = response.choices[0].message.content
        
        # Robust JSON extraction
        start_idx = result_text.find('{')
        end_idx = result_text.rfind('}')
        if start_idx != -1 and end_idx != -1:
            result_text = result_text[start_idx:end_idx+1]
        
        data = json.loads(result_text)

        rows = []
        for q in data.get("questions", []):
            q_type = q.get("question_type", "mcq").lower()
            q_type = TYPE_MAP.get(q_type, q_type)
            rows.append({
                "question": q.get("question", ""),
                "question_type": q_type,
                "options": "|".join(q.get("options", [])),
                "correct_answers": "|".join(q.get("correct_answers", [])),
                "explanation": q.get("explanation", ""),
                "tags": "|".join(q.get("tags", [])),
            })
        print(f"[{start_min}m-{end_min}m] Successfully parsed {len(rows)} questions.")
        return rows
    except Exception as e:
        print(f"[{start_min}m-{end_min}m] Error: {e}", file=sys.stderr)
        try:
            print(f"[{start_min}m-{end_min}m] Raw response snippet: {result_text[:500]}", file=sys.stderr)
        except Exception:
            pass
        return []

async def main():
    parser = argparse.ArgumentParser(description="Efficient YouTube-to-Quiz via OpenRouter")
    parser.add_argument("url")
    parser.add_argument("--api-key", default=os.environ.get("OPENROUTER_API_KEY"))
    parser.add_argument("--qpm", type=int, default=3, help="Questions per minute")
    parser.add_argument("--output", default="quiz.csv")
    parser.add_argument("--batch-minutes", type=int, default=5, help="Minutes per chunk")
    args = parser.parse_args()

    if not args.api_key:
        print("Set OPENROUTER_API_KEY environment variable."); sys.exit(1)

    client = OpenRouter(api_key=args.api_key)

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
            asyncio.to_thread(
                generate_questions_for_chunk,
                client,
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
            fieldnames=["question", "question_type", "options", "correct_answers", "explanation", "tags"],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Done! Generated {len(all_rows)} questions and wrote to {args.output}.")

if __name__ == "__main__":
    asyncio.run(main())