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
  ┌─────────────────┐
  │ Privacy Scrubber│  ← regex masks API keys, tokens, PII
  └──────┬──────────┘
         │
         ▼
  ┌─────────────┐
  │  Compressor │  ← rule-based extraction + LLM fallback
  │  (20 words) │
  └──────┬──────┘
         │
         ▼
  ┌──────────────────┐
  │  Semantic Dedup  │  ← cosine sim check via sqlite-vec
  │  discard if >0.95│    or strengthen existing insight weight
  └──────┬───────────┘
         │
         ▼
  ┌─────────────────┐
  │  Quality Filter │  ← scores against 9-book framework signals
  │  threshold ≥ 15 │    + temporal decay on older insights
  └──────┬──────────┘
         │ pass          │ fail → discard
         ▼
  ┌─────────────────┐
  │   SQLite DB     │  insights.db (~/.grow-ai/)
  │ (insight + full │  stores BOTH compressed + full context
  │  context)       │
  └──────┬──────────┘
         │
    ┌────┴────┐
    ▼         ▼
sqlite-vec  Fine-tune batch
 (RAG)     (every 50 insights OR 7 days)
    │              │
    ▼              ▼
Immediate      LoRA training (Unsloth)
retrieval      (RTX 5070 Ti)
                   │
                   ▼
            Growth Log benchmark
                   │
                   ▼
            Ollama model reload
```

---

## 5. Data Pipeline Detail

### 5.0 Privacy Scrubber (NEW — runs before everything)
Deterministic regex pass before any compression or storage:
- Masks: API keys, tokens, passwords, env vars, IP addresses, emails, file paths with credentials
- Pattern: `sk-[a-zA-Z0-9]{32,}` → `[API_KEY]`, `Bearer [token]` → `[AUTH_TOKEN]`
- Zero LLM calls — pure regex, <5ms
- Anything matching scrubber patterns is replaced inline before the insight hits SQLite

### 5.1 Hook — Capture
- **Event**: `PostToolUse` (captures tool name, input, output, success/fail)
- **Delta-of-Success detection**: If a tool fails and then succeeds within 3 turns → flag as `error_recovery=true` → +20 quality score (Antifragile signal)
- **Sliding window**: Last 10 messages as context
- **Format captured**: `{ tool, action, result, timestamp, session_id, error_recovery }`

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

### 5.2b Semantic De-duplication (NEW — runs after compression)
Before writing to SQLite, check cosine similarity against existing insights via sqlite-vec:
- If similarity > 0.95 → **discard** new insight, increment `reinforcement_count` on existing row
- If 0.80–0.95 → **merge** (update existing insight's `quality_score += 3`)
- If < 0.80 → **store** as new insight
- Prevents data noise from working on the same bug for hours

### 5.3 Quality Scoring
Each insight is scored against signal weights:

| Signal | Weight |
|--------|--------|
| Tool type (Edit/Write > Read) | +10 |
| Error recovery (fixed a bug) | +15 |
| Delta-of-Success (fail → fix within 3 turns) | +20 |
| New file created | +12 |
| Output size (>50 lines) | +5 |
| Framework keyword detected | +8 per match |
| Repeated/boilerplate action | -10 |

**Temporal decay**: Quality score of stored insights decays by 2% per week. Insights > 6 months old have ~50% influence on LoRA vs. recent ones. Allows model to "grow out of" old habits naturally.

**Threshold**: Score ≥ 15 → store. Score < 15 → discard.

### 5.4 Storage Schema (SQLite)
```sql
-- insights table
CREATE TABLE insights (
  id INTEGER PRIMARY KEY,
  compressed TEXT NOT NULL,        -- the 20-word insight
  full_context TEXT NOT NULL,      -- original capture (always stored — disk is cheap)
  framework_tags TEXT,             -- JSON array: ["atomic_habits", "make_it_stick"]
  quality_score INTEGER,
  reinforcement_count INTEGER DEFAULT 0,  -- how many deduped near-duplicates merged in
  error_recovery INTEGER DEFAULT 0,       -- 1 if this was a fail→fix insight
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

-- growth_log table (benchmarks)
CREATE TABLE growth_log (
  id INTEGER PRIMARY KEY,
  insight_count INTEGER,           -- total insights at time of benchmark
  model_version TEXT,
  question_id INTEGER,             -- 1-5 (the 5 benchmark questions)
  response TEXT,                   -- model's answer
  recorded_at DATETIME DEFAULT CURRENT_TIMESTAMP
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

**Upgrade process (v1 — simple full re-train, no distillation)**:
1. Export full accumulated SQLite insights dataset as JSONL
2. Fine-tune new base model (e.g. 7B) directly on the full dataset via Unsloth
3. Deploy on Ollama, retire old model
4. The insights dataset IS the knowledge transfer — no distillation complexity needed in v1

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

## 8b. Growth Log — Proving the Model Improves

Every 50 insights, the system automatically asks the model the same 5 benchmark questions and logs the response to `growth_log`:

1. "How would you design a feedback loop for a React state manager?"
2. "You have 3 hours to debug an unknown production error. Walk me through your approach."
3. "How do you decide when to stop exploring solutions and commit to one?"
4. "Design a personal learning system for mastering a new programming language in 30 days."
5. "A habit you built 6 months ago is breaking down. What's your diagnosis and fix?"

Responses are stored verbatim. Users can scroll a timeline and see the model evolve from generic LLM output to framework-grounded reasoning. This is the **star-worthy demo**.

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
- **9-Book Reasoning Gold Set** — release the JSONL training dataset as a standalone asset. High-quality reasoning data is rare; community will build ecosystem around it, driving stars independently
- Curated insights quality takes months to build, can't be cloned
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

| Component | Technology | Decision |
|-----------|-----------|----------|
| Base model | Qwen2.5-3B (Ollama) | — |
| Fine-tuning | Unsloth + LoRA | Unsloth: best memory efficiency on RTX 5070 Ti, handles 7B/14B without OOM |
| Storage | SQLite (insights + vectors) | sqlite-vec extension for RAG — single file, simple backup, local-first |
| Hook system | Claude Code `PostToolUse` hook (Python) | — |
| Privacy scrubber | Python regex (pre-compression) | No LLM calls, <5ms |
| Compression | Rule-based → Qwen2.5-3B fallback | — |
| Semantic dedup | sqlite-vec cosine similarity | Same DB, no extra process |
| Training runtime | Python + Unsloth + transformers | — |
| Hardware | RTX 5070 Ti (16GB VRAM) | — |
| API server | FastAPI | — |
| Frontend (Phase 2) | Next.js dashboard | — |

---

## 11. Out of Scope (v1)

- Multi-user support
- Cloud training (Phase 2 only)
- Marketplace UI (Phase 2 only)
- Voice interface
- Mobile app

---

## 12. Resolved Decisions

| Question | Decision | Reason |
|----------|----------|--------|
| Store full context or just compressed? | **Both** | Disk is cheap; full context needed to regenerate better instruction pairs if scoring logic changes later |
| Unsloth vs LLaMA-Factory? | **Unsloth** | Better memory efficiency on RTX 5070 Ti; handles 7B/14B with larger context without OOM |
| ChromaDB vs sqlite-vec? | **sqlite-vec** | Single-file stack, trivial backup, fits local-first philosophy |
| Knowledge distillation on upgrade? | **No — full re-train on new base** | Simpler for v1; accumulated JSONL dataset IS the knowledge transfer |
