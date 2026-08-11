# YouTube Transcript → MCQ Generator

Automatically fetches a YouTube video's transcript and generates a rich set of quiz questions (MCQ, multi-select, fill-in-the-blank, and flashcards) using OpenRouter LLMs. Output is saved as a ready-to-import CSV.

---

## Prerequisites

- Python 3.9+
- A virtual environment (recommended)
- An [OpenRouter](https://openrouter.ai/) API key

---

## Installation

```bash
# 1. Clone the repo
git clone <repo-url>
cd fastapi_youtube_transcript_mcq-s

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install yt-dlp openrouter python-dotenv pydantic
```

---

## Configuration

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxx
```

> The script loads this file automatically via `python-dotenv`. You can also pass the key directly with `--api-key`.

---

## Usage

```bash
python main.py <YouTube-URL> [options]
```

### Basic example

```bash
python main.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
```

### All options

| Argument | Default | Description |
|---|---|---|
| `url` | *(required)* | YouTube video URL |
| `--api-key` | `$OPENROUTER_API_KEY` | OpenRouter API key |
| `--qpm` | `3` | Questions to generate per minute of video |
| `--batch-minutes` | `5` | Minutes of transcript per processing chunk |
| `--output` | `quiz.csv` | Output CSV file path |

### Examples

```bash
# Custom output file and 5 questions per minute
python main.py "https://youtu.be/VIDEO_ID" --qpm 5 --output my_quiz.csv

# Explicit API key (no .env needed)
python main.py "https://youtu.be/VIDEO_ID" --api-key sk-or-v1-xxx

# Larger chunks for shorter videos
python main.py "https://youtu.be/VIDEO_ID" --batch-minutes 10
```

---

## Output Format

The generated `quiz.csv` contains the following columns:

| Column | Description |
|---|---|
| `question` | The question text |
| `question_type` | One of: `mcq`, `multi_select`, `fill_blank`, `flashcard` |
| `options` | Pipe-separated answer options (empty for fill_blank/flashcard) |
| `correct_answers` | Pipe-separated correct answer(s) |
| `explanation` | Explanation of the correct answer |
| `tags` | Pipe-separated topic tags |

---

## How It Works

1. **Fetch transcript** – `yt-dlp` downloads the English subtitles from YouTube.
2. **Chunk transcript** – The transcript is split into configurable time windows (default: 5 min each).
3. **Generate questions** – Each chunk is sent concurrently to OpenRouter's LLM API.
4. **Write CSV** – All questions are merged and saved to the output file.