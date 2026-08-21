# Commit Message Convention for Scaling-Law Snapshots

> This is the commit convention the **coordinator** uses after each chunk of 5 tasks. Retires the per-task commit convention used outside scaling-law mode.

---

## When to Commit

- **After all 5 subagents in the current chunk have returned** (success, BLOCKED, or 90-min timeout).
- **After the coordinator has reviewed all 5 `findings.md` files and updated `.claude/libero/skills/` accordingly.**
- **Never during a chunk** — mid-chunk skill edits would race with in-flight subagent reads.

One commit per chunk. Eighteen total for a full scaling run.

---

## Required Fields in the Commit Message

```
Snapshot N=<N>: tasks <t1> <t2> <t3> <t4> <t5>

Tasks solved:  <n_solved>/5  (<n_blocked> blocked after 30 seeds)

Tokens (chunk build, deduplicated by requestId):
  input:    <input_tokens>
  cache:    <cache_tokens>
  output:   <output_tokens>
  total:    <total_tokens>
Tokens (cumulative through N=<N>):
  total:    <cumulative_total>

Skill updates:
  localize:      +<loc_lines>L
  grasp:         +<grasp_lines>L
  transport:     +<transport_lines>L
  manipulation:  +<manip_lines>L

Summary: <one-line distilled lesson from this chunk>

Refs: ordering.txt, tag snapshot-N<N>
```

- `<N>` = chunk_index × 5 (5, 10, 15, ..., 90)
- Task names come from `ordering.txt` lines `5*(chunk_index-1)+1` through `5*chunk_index`
- Token counts come from the `/token-calculate` skill
- Skill line counts via `git diff --stat snapshot-N<prev>..HEAD -- .claude/libero/skills/`

---

## Example

```
Snapshot N=25: tasks KITCHEN_SCENE6_put_the_yellow_and_white_mug_on_the_plate \
  LIVING_ROOM_SCENE2_pick_up_the_ketchup_and_put_it_in_the_basket \
  STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy \
  KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it \
  KITCHEN_SCENE8_open_the_microwave

Tasks solved:  4/5  (1 blocked after 30 seeds)

Tokens (chunk build, deduplicated by requestId):
  input:    198432
  cache:    3847212
  output:   67349
  total:    4113093
Tokens (cumulative through N=25):
  total:    19847203

Skill updates:
  localize:      +12L   (added "ketchup" prompt, "book" disambiguation by bbox)
  grasp:         +0L
  transport:     +5L    (basket rim-clearance waypoint)
  manipulation:  +18L   (stove knob CCW 120° tuning, moka pot placement)

Summary: Basket rim-clearance matters for caddy compartments too.
         Stove knob turn-on needs 120° not 90° on some knobs.

Refs: ordering.txt, tag snapshot-N25
```

---

## Post-Commit: Tag

Immediately after the commit:

```bash
git tag "snapshot-N${N}"
```

The tag is immutable; never re-tag. Eval orchestration checks out this tag in a worktree to freeze the library state during eval.

---

## If Nothing Changed in `.claude/libero/skills/`

Still commit (so the snapshot is git-contiguous) but with a brief message:

```
Snapshot N=<N>: tasks <t1>..<t5>

Tasks solved:  <n>/5
Skill updates: none (library unchanged from N=<N-5>)

Tokens (chunk build):  <...>
Tokens (cumulative):   <...>

Refs: ordering.txt, tag snapshot-N<N>
```

Then tag as usual. Eval still runs against this snapshot.

---

## What NOT to Include

- ❌ Personal notes / speculation. Keep it terse and factual.
- ❌ References to tasks outside this chunk.
- ❌ `git push` commands — never push (per `CLAUDE.md`).
- ❌ Co-authored-by tags unless you've verified Git config allows it without identity leakage.
