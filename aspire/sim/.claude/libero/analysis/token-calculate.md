# Token Usage Analysis Skill

Scripts and patterns for parsing Claude Code JSONL conversation files to compute token usage over time.

---

## Data Source

Claude Code stores every conversation as a JSONL file under:
```
~/.claude/projects/<sanitized-cwd>/
  <session-uuid>.jsonl          # main conversations
  <session-uuid>/subagents/     # subagent conversations spawned via Agent tool
```

Each `assistant`-type line contains a `message.usage` block:
```json
{
  "type": "assistant",
  "timestamp": "2026-04-22T14:03:12.000Z",
  "requestId": "req_abc123",
  "isSidechain": false,
  "message": {
    "model": "claude-sonnet-4-6",
    "usage": {
      "input_tokens": 4,
      "cache_creation_input_tokens": 2971,
      "cache_read_input_tokens": 110920,
      "output_tokens": 208
    }
  }
}
```

Token fields:
- `input_tokens` — fresh uncached input
- `cache_creation_input_tokens` — tokens written to prompt cache
- `cache_read_input_tokens` — tokens read from prompt cache
- `output_tokens` — generated output

---

## Critical: Deduplication

**Sidechain messages are stored twice**: once as `isSidechain: true` entries inside the parent conversation JSONL, and again in the subagent's own JSONL file. Counting both inflates totals ~2-3×.

**Fix**: deduplicate by `requestId` across all files:

```python
seen_requests: set[str] = set()

for f in Path("~/.claude/projects").rglob("*.jsonl"):
    for line in f.read_text(errors="replace").splitlines():
        d = json.loads(line)
        if d.get("type") != "assistant": continue
        req_id = d.get("requestId", "")
        if req_id:
            if req_id in seen_requests:
                continue          # skip duplicate
            seen_requests.add(req_id)
        # accumulate usage ...
```

This gives one count per actual API call, including subagents, without double-counting.

---

## Scripts

### `scripts/common/plot_token_usage.py`
Reads all JSONL files, deduplicates, produces:
- `outputs/plots/token_usage.png` — stacked bar: daily cost by model (Opus/Sonnet/Haiku)
- Console table: per-day breakdown

Run: `.venv/bin/python3 scripts/common/plot_token_usage.py`

### `scripts/common/plot_subagent_token_trace.py`
Produces:
- `outputs/plots/token_usage_lines2.png` — daily token counts as lines (all models merged); dual-axis since cache_read (billions) dwarfs output/input/cache_write (millions)
- `outputs/plots/subagent_token_trace.png` — per-turn token trace for 3 representative Evolutionary Search debug subagents, showing how cache_read grows monotonically as conversation history accumulates and cache_write spikes on 5-min TTL renewals

Run: `.venv/bin/python3 scripts/common/plot_subagent_token_trace.py`
