#!/usr/bin/env python3
"""Generate a quiz CSV from a local transcript file (no YouTube download).

Expected transcript format (one cue per line):

    [00:00:13] Hello and welcome to week four
    [00:00:15] uh course on machine learning

`[MM:SS]`, `[HH:MM:SS]` and `[HH:MM:SS.mmm]` are all accepted. Lines without a
leading timestamp are appended to the previous cue.
"""
import argparse
import asyncio
import csv
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openrouter import OpenRouter
from openrouter import errors as or_errors

load_dotenv()  # Load environment variables from .env file

DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"

# Retrying these is pointless - a bad key or a malformed request fails identically
# every time. Everything else (502/overloaded, rate limits, timeouts, and the
# ResponseValidationError the SDK raises when a provider error body arrives in
# place of a completion) is treated as transient and retried with backoff.
NON_RETRYABLE_ERRORS = tuple(
    getattr(or_errors, name)
    for name in (
        "UnauthorizedResponseError",
        "PaymentRequiredResponseError",
        "ForbiddenResponseError",
        "BadRequestResponseError",
        "NotFoundResponseError",
        "UnprocessableEntityResponseError",
    )
    if hasattr(or_errors, name)
)

TIMESTAMP_RE = re.compile(
    r"^\s*[\[\(]?\s*(?:(\d+):)?(\d{1,2}):(\d{2})(?:[.,](\d{1,3}))?\s*[\]\)]?\s*(.*)$"
)


class TranscriptUnavailable(Exception):
    pass


# --- Parsing ---

def parse_timestamp_line(line: str):
    """Return (start_seconds, text) for a timestamped line, else None."""
    match = TIMESTAMP_RE.match(line)
    if not match:
        return None
    hours, minutes, seconds, millis, text = match.groups()
    # A bare "12:34" is MM:SS, not HH:MM.
    total = int(minutes) * 60 + int(seconds)
    if hours is not None:
        total += int(hours) * 3600
    if millis:
        total += int(millis.ljust(3, "0")) / 1000.0
    return total, text.strip()


def load_transcript(path: Path) -> List[Dict]:
    """Read a transcript file into [{"start": seconds, "text": str}, ...]."""
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise TranscriptUnavailable(f"Transcript file not found: {path}") from exc
    except OSError as exc:
        raise TranscriptUnavailable(f"Could not read {path}: {exc}") from exc

    entries: List[Dict] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parsed = parse_timestamp_line(stripped)
        if parsed is None:
            # Continuation of the previous cue (wrapped line).
            if entries:
                entries[-1]["text"] = f"{entries[-1]['text']} {stripped}".strip()
            continue
        start, text = parsed
        if text:
            entries.append({"start": start, "text": text})

    if not entries:
        raise TranscriptUnavailable(
            f"No timestamped lines found in {path}. Expected lines like '[00:01:23] some text'."
        )

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


def format_timestamp(total_minutes: int) -> str:
    """Minute index -> HH:MM:SS."""
    hours, minutes = divmod(int(total_minutes), 60)
    return f"{hours:02d}:{minutes:02d}:00"


# --- Question generation ---

TYPE_MAP = {
    "mcq": "mcq",
    "multi_select": "multi_select",
    "fill_blank": "fill_blank",
    "flashcard": "flashcard",
}


def summarize_error(exc: Exception, limit: int = 180) -> str:
    """Collapse a multi-line SDK/pydantic error into one readable line."""
    try:
        text = " ".join(str(exc).split())
    except Exception:
        # Some SDK error types raise from __str__ when partially constructed.
        text = ""
    if len(text) > limit:
        text = text[:limit] + "..."
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def call_model_with_retry(client, model: str, prompt: str, label: str, retries: int, timeout_ms: int) -> str:
    """Send one prompt, retrying transient provider failures with exponential backoff."""
    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            response = client.chat.send(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                timeout_ms=timeout_ms,
            )
            return response.choices[0].message.content
        except NON_RETRYABLE_ERRORS:
            raise
        except Exception as exc:
            last_exc = exc
            if attempt == retries:
                break
            # 2s, 4s, 8s... with jitter so retried chunks don't resynchronize.
            delay = (2 ** attempt) * random.uniform(0.7, 1.3)
            print(
                f"{label} attempt {attempt}/{retries} failed ({summarize_error(exc)}); "
                f"retrying in {delay:.1f}s",
                file=sys.stderr,
            )
            time.sleep(delay)
    raise last_exc  # type: ignore[misc]


def generate_questions_for_chunk(
    client,
    model: str,
    chunk_text: str,
    start_min: int,
    end_min: int,
    q_per_min: int,
    retries: int,
    timeout_ms: int,
) -> Tuple[List[Dict], Optional[str]]:
    """Generate questions for one transcript chunk. Returns (rows, error_message)."""
    num_questions = q_per_min * max(1, end_min - start_min)
    label = f"[{start_min}m-{end_min}m]"
    print(f"{label} Requesting {num_questions} questions from OpenRouter...")

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

    result_text = ""
    try:
        result_text = call_model_with_retry(client, model, prompt, label, retries, timeout_ms)

        # Robust JSON extraction
        start_idx = result_text.find('{')
        end_idx = result_text.rfind('}')
        if start_idx != -1 and end_idx != -1:
            result_text = result_text[start_idx:end_idx + 1]

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
                "start_time": format_timestamp(start_min),
                "end_time": format_timestamp(end_min),
                "start_seconds": start_min * 60,
            })
        print(f"{label} Successfully parsed {len(rows)} questions.")
        return rows, None
    except Exception as e:
        message = summarize_error(e)
        print(f"{label} Giving up: {message}", file=sys.stderr)
        if result_text:
            print(f"{label} Raw response snippet: {result_text[:500]}", file=sys.stderr)
        return [], message


def resolve_output_path(output: str, stamp_filename: bool) -> Path:
    path = Path(output)
    if stamp_filename:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = path.with_name(f"{path.stem}_{stamp}{path.suffix or '.csv'}")
    return path


async def main():
    parser = argparse.ArgumentParser(
        description="Generate a timestamped quiz CSV from a local transcript file."
    )
    parser.add_argument("transcript", help="Path to the transcript file (e.g. transcripts.txt)")
    parser.add_argument("--api-key", default=os.environ.get("OPENROUTER_API_KEY"))
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenRouter model id")
    parser.add_argument("--qpm", type=int, default=3, help="Questions per minute")
    parser.add_argument("--output", default="quiz.csv")
    parser.add_argument("--batch-minutes", type=int, default=5, help="Minutes per chunk")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="Chunks in flight at once. Free models overload easily; keep this low.",
    )
    parser.add_argument("--retries", type=int, default=4, help="Attempts per chunk before giving up")
    parser.add_argument(
        "--timeout", type=float, default=180.0, help="Per-request timeout in seconds"
    )
    parser.add_argument(
        "--timestamp-filename",
        action="store_true",
        help="Append a run timestamp to the output filename (quiz_20260902_101500.csv)",
    )
    args = parser.parse_args()

    if not args.api_key:
        print("Set OPENROUTER_API_KEY environment variable.")
        sys.exit(1)

    client = OpenRouter(api_key=args.api_key)

    print(f"Reading transcript from {args.transcript}...")
    try:
        entries = load_transcript(Path(args.transcript))
    except TranscriptUnavailable as e:
        print(str(e))
        sys.exit(1)

    duration_min = entries[-1]["start"] / 60
    print(f"Parsed {len(entries)} transcript lines covering ~{duration_min:.1f} minutes.")

    chunks = build_transcript_chunks(entries, batch_size_minutes=args.batch_minutes)
    if not chunks:
        print("No transcript text found after parsing.")
        sys.exit(1)

    # Results accumulate here as each chunk finishes, so an interrupt or a failed
    # chunk still leaves us with everything that did succeed.
    all_rows: List[Dict] = []
    failures: List[str] = []
    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    timeout_ms = int(args.timeout * 1000)

    async def run_chunk(chunk: Dict) -> None:
        async with semaphore:
            rows, error = await asyncio.to_thread(
                generate_questions_for_chunk,
                client,
                args.model,
                chunk["text"],
                chunk["start_min"],
                chunk["end_min"],
                args.qpm,
                args.retries,
                timeout_ms,
            )
            all_rows.extend(rows)
            if error:
                failures.append(
                    f"{format_timestamp(chunk['start_min'])}-{format_timestamp(chunk['end_min'])}: {error}"
                )

    print(f"Processing {len(chunks)} batches, {args.concurrency} at a time...")
    try:
        await asyncio.gather(*(run_chunk(chunk) for chunk in chunks))
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\nInterrupted - writing whatever finished so far...", file=sys.stderr)

    if not all_rows:
        print("No questions were generated.")
        if failures:
            print("Failures:", file=sys.stderr)
            for failure in failures:
                print(f"  {failure}", file=sys.stderr)
        sys.exit(1)

    all_rows.sort(key=lambda row: row["start_seconds"])

    output_path = resolve_output_path(args.output, args.timestamp_filename)
    with open(output_path, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "start_time",
                "end_time",
                "start_seconds",
                "question",
                "question_type",
                "options",
                "correct_answers",
                "explanation",
                "tags",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Done! Generated {len(all_rows)} questions and wrote to {output_path}.")
    if failures:
        print(f"\n{len(failures)} of {len(chunks)} segments produced no questions:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        print("Re-run with a higher --retries, a lower --concurrency, or a paid --model.", file=sys.stderr)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
