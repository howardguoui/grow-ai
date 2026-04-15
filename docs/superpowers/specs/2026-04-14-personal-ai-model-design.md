# grow-ai — Design Spec
**Date**: 2026-04-14  
**Status**: Draft — awaiting user approval  
**Project**: `E:\ClaudeProject\grow-ai`  
**License**: MIT (fully open source, free to self-host)

---

## 1. Vision

An open-source, locally-running personal AI model that grows with its owner. Every Claude Code session automatically feeds compressed insights into the model. The model internalizes 9 book frameworks as its cognitive operating system — making it think, remember, and reason in a specific, personal way. Over time it matures from a 3B model to up to 18B.

**Goal**: Become the best personal AI growth tool on GitHub. Community adoption first, hosted platform revenue second.

---

## 2. Core Principles

- **Local-first**: Runs on Ollama, no data leaves your machine
- **Automatic**: No manual curation — hooks capture everything
- **Smart filtering**: Like the human hippocampus, only meaningful signals get encoded
- **20-word compression**: Every captured insight is distilled to its essence before storage
- **Framework-grounded**: 9 books define how the model thinks, not just what it knows

---

## 3. The 9-Book Cognitive Framework

The model's reasoning is grounded in 4 domains:

### Thinking & Logic
| Book | Core Framework | Key Signal |
|------|---------------|------------|
| Thinking Fast and Slow | System 1 (fast/intuitive) vs System 2 (slow/logical) | bias, shortcut, deliberate |
| Clear Thinking | Controlling default reactions in ordinary moments | reaction, default, override |
| Algorithms to Live By | Computer science heuristics applied to life decisions | sort, cache, search, optimize |

### Memory & Study
| Book | Core Framework | Key Signal |
|------|---------------|------------|
| Moonwalking with Einstein | Memory Palace — visualization + association | visualize, link, palace, encode |
| The Memory Book | Logical systems for names, numbers, lists | hook, peg, chain, anchor |

### Learning & Mastery
| Book | Core Framework | Key Signal |
|------|---------------|------------|
| Make It Stick | Active recall, spaced repetition, interleaving | recall, practice, spacing, test |
| Ultralearning | Intense self-directed skill acquisition via focused projects | drill, feedback, directness |

### Systems & Evolution
| Book | Core Framework | Key Signal |
|------|---------------|------------|
| Atomic Habits | Habit loop: cue → craving → response → reward | habit, cue, trigger, streak |
| Thinking in Systems | Patterns over events, feedback loops, leverage points | loop, pattern, system, flow |
| Antifragile | Growing stronger through volatility and stress | stress, volatility, resilience |

---

## 4. Architecture Overview

```
Claude Code Session
        │
        ▼ (PostToolUse hook)
  ┌─────────────┐
  │  Compressor │  ← rule-based extraction + LLM fallback
  │  (20 words) │
  └──────┬──────┘
         │
         ▼
  ┌─────────────────┐
  │  Quality Filter │  ← scores against 9-book framework signals
  │  threshold ≥ 15 │
  └──────┬──────────┘
         │ pass          │ fail → discard
         ▼
  ┌─────────────┐
  │   SQLite DB │  insights.db
  │  (~/.claude-ai/)│
  └──────┬──────┘
         │
    ┌────┴────┐
    ▼         ▼
  RAG DB    Fine-tune batch
(ChromaDB)  (every 50 insights OR 7 days)
    │              │
    ▼              ▼
Immediate      LoRA training
retrieval      (RTX 5070 Ti)
                   │
                   ▼
            Ollama model reload
```

---

## 5. Data Pipeline Detail

### 5.1 Hook — Capture
- **Event**: `PostToolUse` (captures tool name, input, output, success/fail)
- **Sliding window**: Last 10 messages as context
- **Format captured**: `{ tool, action, result, timestamp, session_id }`

### 5.2 Compression — 20-Word Insight
Two-stage process:

**Stage 1 — Rule-based** (<100ms, free):
```
[Tool: {tool}] [Action: {verb + object}] [Result: {outcome}]
Example: "Edit src/app.js: Added React auth hook — token stored, session persists"
```

**Stage 2 — LLM fallback** (~300ms, Qwen2.5-3B):
- Only fires if Stage 1 produces generic/low-signal output
- Prompt: "Summarize the key learning from this interaction in ≤20 words"

### 5.3 Quality Scoring
Each insight is scored against signal weights:

| Signal | Weight |
|--------|--------|
| Tool type (Edit/Write > Read) | +10 |
| Error recovery (fixed a bug) | +15 |
| New file created | +12 |
| Output size (>50 lines) | +5 |
| Framework keyword detected | +8 per match |
| Repeated/boilerplate action | -10 |

**Threshold**: Score ≥ 15 → store. Score < 15 → discard.

### 5.4 Storage Schema (SQLite)
```sql
-- insights table
CREATE TABLE insights (
  id INTEGER PRIMARY KEY,
  compressed TEXT NOT NULL,        -- the 20-word insight
  full_context TEXT,               -- original capture (for retraining)
  framework_tags TEXT,             -- JSON array: ["atomic_habits", "make_it_stick"]
  quality_score INTEGER,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- fine_tune_batches table
CREATE TABLE fine_tune_batches (
  id INTEGER PRIMARY KEY,
  insight_ids TEXT,                -- JSON array of insight IDs in this batch
  triggered_at DATETIME,
  completed_at DATETIME,
  model_version TEXT,
  status TEXT                      -- queued, running, done, failed
);
```

### 5.5 Fine-Tune Trigger
Trigger LoRA training when ANY of:
- 50 new insights accumulated since last run
- 7 days elapsed since last run
- Manual `/finetune` command

### 5.6 Training Format
Insights are expanded to 50-100 word instruction pairs for LoRA:
```json
{
  "instruction": "How should I approach debugging a React hook that causes infinite re-renders?",
  "input": "",
  "output": "Apply Systems Thinking: identify the feedback loop first. Map what triggers the effect, what the effect changes, and whether that change re-triggers. Use spaced debugging — add one console.log at a time and let the pattern emerge before fixing."
}
```

---

## 6. Model Growth Path

The model cannot literally grow — it upgrades to a larger base at milestones:

| Phase | Base Model | When | VRAM |
|-------|-----------|------|------|
| 0 | Qwen2.5-3B | Start | ~4GB |
| 1 | Qwen2.5-7B | ~3-4 months daily use | ~8GB |
| 2 | Qwen2.5-14B | ~6-9 months | ~12GB |
| 3 | Qwen2.5-18B | ~12+ months | ~16GB |

**Upgrade process**:
1. Merge current LoRA adapter into old base → export merged weights
2. Knowledge-distill merged model onto new base (transfer learning)
3. Train fresh LoRA adapter on new base
4. Deploy on Ollama, retire old model

**Catastrophic forgetting prevention**: Replay buffer — every LoRA training run uses 50% historical insights + 50% new insights.

---

## 7. Framework-Aware Response

When user sends a prompt, the model:
1. Scores the prompt against all 9 frameworks
2. Activates top 2-3 frameworks (score threshold > 40)
3. Adapts response mode:

| Mode | Frameworks | Behavior |
|------|-----------|---------|
| Diagnostic | Kahneman, Parrish | Identify System 1 traps, default reactions |
| Prescriptive | Atomic Habits, Make It Stick | Give actionable steps, habit design |
| Exploratory | Algorithms to Live By, Ultralearning | Explore solution space systematically |
| Structural | Thinking in Systems, Antifragile | Map patterns, feedback loops, stressors |

---

## 8. Compressed Insight Examples

| Framework | Example 20-word insight |
|-----------|------------------------|
| Thinking Fast and Slow | "Chose first solution that worked — System 1 override. Next time map 3 options before committing." |
| Algorithms to Live By | "Sorted debug candidates by most-recently-changed files first — optimal explore-exploit for bug hunting." |
| Memory Palace | "Linked JWT flow to airport security analogy — baggage=payload, boarding pass=token, gate=verify." |
| Make It Stick | "Third async/await attempt this week — spaced retrieval practice compressing the gap noticeably." |
| Atomic Habits | "Daily commit streak hit 14 days — cue: end-of-session, reward: green square. Compound effect visible." |
| Thinking in Systems | "Auth bug traced to feedback loop: bad token → retry → flood → rate-limit → more retries." |
| Antifragile | "Deploy failed in prod — forced better error handling. System stronger from the stress." |

---

## 9. GitHub-First Growth Strategy

### Why open source wins here
- Novel concept ("AI that reads books and learns from your work sessions") → viral on GitHub, HN, Reddit
- Free to self-host → zero friction adoption → stars accumulate
- Community contributes new book framework packs → moat deepens
- Reference: `virattt/ai-hedge-fund` — same model, thousands of stars

### What makes it star-worthy
1. **The concept is the hook** — "Your AI gets smarter every time you use Claude Code"
2. **Works out of the box** — one command install, runs locally on Ollama
3. **Visible growth** — users can see the model improve (before/after benchmark comparisons)
4. **Book framework packs** — community can add frameworks from other books (plugin system)
5. **Great README** — demo GIF showing model improving over weeks

### Moat (hard to replicate)
- First mover + community = the canonical repo for this concept
- Curated 9-book training dataset (quality takes months to build, can't be cloned)
- Claude Code hook integration maintained as Claude evolves
- Community framework packs accumulate over time

### Monetization (Phase 2 — after community traction)
**Model**: Open core + hosted platform

| Tier | Price | What you get |
|------|-------|-------------|
| Self-host | Free | Full open source, run on your own machine |
| Hosted | $4.99/mo | Deploy on our platform — no GPU, no setup, always on |

**When to launch hosted**: After 500+ GitHub stars (proves demand exists).  
**Target customer**: People who love the concept but don't have a GPU or don't want the setup hassle.

---

## 10. Tech Stack

| Component | Technology |
|-----------|-----------|
| Base model | Qwen2.5-3B (Ollama) |
| Fine-tuning | LoRA via Unsloth or LLaMA-Factory |
| Storage | SQLite (insights) + ChromaDB (RAG vectors) |
| Hook system | Claude Code `PostToolUse` hook (bash script) |
| Compression | Python script + Qwen2.5-3B via Ollama API |
| Training runtime | Python + transformers + peft |
| Hardware | RTX 5070 Ti (16GB VRAM) |
| API server | FastAPI (serve personal model endpoint) |
| Frontend (Phase 2) | Next.js dashboard |

---

## 11. Out of Scope (v1)

- Multi-user support
- Cloud training (Phase 2 only)
- Marketplace UI (Phase 2 only)
- Voice interface
- Mobile app

---

## 12. Open Questions

1. Should Stage 1 compression store the full original context or just the compressed insight? (Storage vs. retraining quality tradeoff)
2. Which LoRA training library — Unsloth (faster) or LLaMA-Factory (more control)?
3. Should the RAG DB be ChromaDB or SQLite-vec (simpler, no extra process)?
