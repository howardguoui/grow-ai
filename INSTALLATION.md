# Grow AI Installation Guide

## Quick Start

### 1. Install Hook (one-time setup)

```bash
python scripts/install_hook.py
```

This idempotent script:
- Creates `~/.claude` directory if needed
- Adds the capture pipeline to PostToolUse hooks in `~/.claude/settings.json`
- Safe to run multiple times

**Options:**
- `--dry-run`: Preview changes without modifying settings
- `--uninstall`: Remove the hook
- `--settings <path>`: Custom settings.json location

### 2. Verify Installation

```bash
# Test the capture pipeline
python -m grow_ai.capture --test

# Or with JSON input
echo '{"text":"How do I learn?","problem_type":"learning"}' | python -m grow_ai.capture
```

## What Gets Installed

The hook installer adds a PostToolUse hook to `~/.claude/settings.json`:

```json
{
  "hooks": {
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
}
```

**When triggered:**
- Every Claude Code tool execution triggers the hook
- Pipeline reads from stdin: `{"text":"...", "problem_type":"...", "domain":"..."}`
- Conversations are scrubbed, compressed, scored, and stored to SQLite
- Results returned as JSON

## Database

Default: `grow_ai.db` in current directory

**Storage contents:**
- Original text
- Scrubbed text (sensitive data removed)
- Compressed summary (20-50 words)
- Vector embedding (768 dimensions)
- Primary framework (from 9 book system)
- Quality score (0-100)
- Timestamp

## Pipeline Stages

1. **Scrub**: Remove sensitive data (API keys, email, PII)
2. **Compress**: Create 20-50 word summary
3. **Embed**: Generate 768-dimension vector (stub, ready for real models)
4. **Dedup**: Check for duplicates (threshold: 0.85)
5. **Score**: Evaluate against 9 book frameworks (threshold: 30/100)
6. **Store**: Save to SQLite database

## Testing

```bash
# All tests
pytest tests/ -v

# Specific test
pytest tests/test_capture.py -v

# With coverage
pytest tests/ --cov=grow_ai
```

**Current status: 60/60 tests passing**

## Uninstall

```bash
python scripts/install_hook.py --uninstall
```

This safely removes the hook while preserving other hooks in settings.json.

## Troubleshooting

**"Module not found" error:**
```bash
pip install -e .
```

**Database locked error:**
- The database may be in use by another process
- Ensure only one instance of Claude Code is running
- Or specify a different database: `python -m grow_ai.capture --db custom.db`

**Hook not executing:**
- Verify settings.json was updated: `cat ~/.claude/settings.json | grep grow_ai`
- Check stderr for errors: `python scripts/install_hook.py --dry-run`
- May require Claude Code restart to pick up hook changes

## CLI Reference

### Capture Pipeline

```bash
python -m grow_ai.capture [--db DB] [--dry-run] [--test]

Options:
  --db DB        Path to SQLite database (default: grow_ai.db)
  --dry-run      Process without storing
  --test         Run with example data
```

### Hook Installer

```bash
python scripts/install_hook.py [--dry-run] [--settings PATH] [--uninstall]

Options:
  --dry-run      Preview changes
  --settings     Custom settings.json path
  --uninstall    Remove hook
```

## Architecture

```
Claude Code Session
        ↓
   PostToolUse Hook (triggered after each tool)
        ↓
grow_ai.capture (stdin JSON input)
        ↓
[Scrub] → [Compress] → [Embed] → [Dedup] → [Score] → [Store]
        ↓
SQLite Database (grow_ai.db)
        ↓
JSON Output (status, all stages, stored_id)
```

## Next Steps

- Implement real embeddings (currently stub: all zeros)
- Integrate with embedding models (local or API)
- Build retrieval/similarity search from embeddings
- Add fine-tuning loop for personal model
