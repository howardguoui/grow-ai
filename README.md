# grow-ai

A self-learning knowledge capture system that passively records insights from your Claude Code sessions, deduplicates them semantically, and prepares LoRA fine-tune batches — all locally, zero API cost.

## How it works

```
Claude Code session
       │
       ├─ PostToolUse ──► grow_ai.capture       (every tool call)
       │                    scrub → compress → embed → dedup → score → store
       │
       └─ SessionStart ──► session_capture.py   (once at launch)
                            scan yesterday's transcripts
                            YAKE keywords + Sumy LexRank
                            → 3-5 insights per session → embed → store
                                         │
                              ~/.grow-ai/insights.db  (SQLite + vectors)
                                         │
                              python -m grow_ai.daily_routine
                              dedup · decay · expand · finetune · report
```

**Zero paid API calls.** Everything runs locally via Ollama.

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally with:
  - `ollama pull nomic-embed-text` — embeddings
  - `ollama pull qwen2.5:3b` — compression + LoRA pair generation
- Claude Code (for the PostToolUse / SessionStart hooks)

---

## Installation

```bash
git clone https://github.com/howardguoui/grow-ai
cd grow-ai
pip install -e .
pip install sumy yake
```

Download NLTK data (needed by sumy, one-time):

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
```

---

## Claude Code hook setup

Add these to your `~/.claude/settings.json`:

```json
"hooks": {
  "SessionStart": [
    {
      "matcher": "",
      "hooks": [
        {
          "type": "command",
          "command": "python /path/to/.claude/hooks/session_capture.py"
        }
      ]
    }
  ],
  "PostToolUse": [
    {
      "matcher": "",
      "hooks": [
        {
          "type": "command",
          "command": "python -m grow_ai.capture"
        }
      ]
    }
  ]
}
```

Copy the hook scripts from `hooks/` to `~/.claude/hooks/`:

| Script | Hook event | Purpose |
|--------|-----------|---------|
| `session_capture.py` | `SessionStart` | Scans yesterday's transcripts, extracts key insights via YAKE + Sumy |
| (built-in module) | `PostToolUse` | Captures tool interactions in real-time |

---

## Project structure

```
grow-ai/
├── grow_ai/
│   ├── capture.py          # PostToolUse hook entry point
│   ├── compress.py         # 20-word insight compression (rule-based + LLM fallback)
│   ├── config.py           # Ollama URLs, thresholds, paths
│   ├── daily_routine.py    # Maintenance job: dedup, decay, expand, finetune, report
│   ├── db.py               # SQLite schema + helpers
│   ├── dedup.py            # Semantic deduplication via cosine similarity
│   ├── embed.py            # nomic-embed-text via Ollama REST
│   ├── expand.py           # LoRA pair generation via qwen2.5:3b
│   ├── finetune.py         # Fine-tune batch trigger
│   ├── scorer.py           # Quality scoring + framework tag detection
│   ├── scrubber.py         # Secret/PII scrubbing
│   └── search.py           # Semantic + keyword search over stored insights
├── hooks/
│   └── session_capture.py  # SessionStart hook (YAKE + Sumy, zero LLM cost)
├── api/                    # FastAPI REST endpoints
├── tests/
└── pyproject.toml
```

---

## Daily maintenance

Run once a day (or schedule via Task Scheduler / cron):

```bash
cd E:/ClaudeProject/grow-ai
python -m grow_ai.daily_routine
```

Output example:
```
[grow-ai] Dedup: {'merged': 2, 'discarded': 1}
[grow-ai] Decay report: {'total_insights': 47, 'below_threshold_after_decay': 3}
[grow-ai] Expansion: {'processed': 5, 'failed': 0, 'total_unexpanded': 5}
[grow-ai] Fine-tune check: {'triggered': False, 'total_insights': 47}
[grow-ai] Growth report: {'total_insights': 47, 'new_last_24h': 8, 'avg_quality_score': 12.4, ...}
```

Fine-tune batch triggers automatically when 50+ insights accumulate.

---

## Configuration

Edit `grow_ai/config.py`:

```python
ollama_base_url   = "http://localhost:11434"
generative_model  = "qwen2.5:3b"       # for compression + LoRA pairs
embedding_model   = "nomic-embed-text"  # for semantic dedup + search
quality_threshold = 15                  # min score to store tool-use insights
finetune_batch_size = 50               # insights needed to trigger fine-tune
dedup_discard     = 0.95               # similarity → delete duplicate
dedup_merge       = 0.80               # similarity → reinforce + delete
decay_rate_per_week = 0.02             # 2% quality decay per week
```

---

## Data location

All data lives in `~/.grow-ai/`:

```
~/.grow-ai/
├── insights.db    # SQLite: insights + vectors + growth_log + fine_tune_batches
└── state.json     # Last-run state
```

---

## Bugs fixed in this session

| Bug | Fix |
|-----|-----|
| `sqlite_vec.deserialize_float32` missing in v0.1.9 | Replaced with `struct.unpack` in `dedup.py` and `daily_routine.py` |
| `scorer.py` reading `event["tool"]` instead of `event["tool_name"]` | Fixed field name — tool weights now apply correctly |
| `scorer.py` reading `event["result"]` instead of `event["tool_response"]` | Fixed field name |
| `expand.py` Unicode `✔`/`✗` crashing Windows cp1252 terminal | Replaced with ASCII `ok`/`failed` |
| `daily_routine.py` Unicode output crashing on Windows | Added `sys.stdout.reconfigure(encoding="utf-8")` |
| Session capture quality filter blocking all conversation insights | YAKE+Sumy acts as quality gate; score floor applied before insert |
