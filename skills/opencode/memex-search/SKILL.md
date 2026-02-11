---
name: memex-search
description: Search, filter, and retrieve Opencode history via memex CLI. Use for context resumption, finding past code/decisions, and self-correction based on history.
---

# Memex for Opencode

`memex` is the primary memory retrieval tool for historical Opencode sessions.

## Session discovery first

For "which sessions had activity in this time window", start with:

```bash
memex sessions --source opencode --since <iso|unix> --until <iso|unix> --json-array
```

Useful options:
- `--project <name>`
- `--sort last-ts|first-ts|count`
- `--limit <n>`
- `-v/--verbose`

Per-session output includes:
- `session_id`, `project`, `first_ts`, `last_ts`
- `message_count`, `user_count`, `assistant_count`, `tool_use_count`, `tool_result_count`
- `tool_names`, `source_path`, `source_path_exists`

## Usage Patterns

- Context retrieval:
  - `memex search "API discussion" --source opencode --sort ts --limit 10`
- Code discovery:
  - `memex search "function implementation" --source opencode --hybrid`
- Session identification:
  - `memex search "database migration" --source opencode --unique-session`

## Search Modes

| Need | Flag | Example |
| --- | --- | --- |
| Exact terms, IDs, errors | (default) | `memex search "Error: 500" --source opencode` |
| Concepts, intent | `--semantic` | `memex search "auth flow" --source opencode --semantic` |
| Mixed specific + fuzzy | `--hybrid` | `memex search "user_id logic" --source opencode --hybrid` |

If the vector index is unavailable, memex warns on stderr and falls back to lexical search. Treat this as degraded retrieval and mention `memex embed` as the recovery step when useful.

## Background Index Service

- Use `memex index-service enable` to install the background indexer. It runs via launchd on macOS and systemd user services on Linux.
- Use `memex index-service enable --continuous` for a long-lived watcher; add `--poll-interval <seconds>` to tune continuous mode.
- Default mode is periodic indexing, typically every 3600 seconds; add `--interval <seconds>` to tune interval mode.
- The service inherits indexing flags, so pass source and embedding options at install time when needed, e.g. `memex index-service enable --opencode --embeddings`.
- On successful enable, memex writes `auto_index_on_search = false` to config when that setting is absent, so searches do not duplicate daemon work. Explicit user config is preserved.
- macOS writes `~/.memex/index-service.plist`, `~/.memex/index-service.log`, and `~/.memex/index-service.err.log`. Linux writes systemd user units under `~/.config/systemd/user/`.
- Use `memex index-service disable` to unload and remove the service.

## Session Context

Use `--session <session_id>` to isolate one interaction thread.

1. Find the session ID:
   - `memex search "topic" --source opencode --unique-session`
2. Narrow inside that session:
   - `memex search "detail" --source opencode --session <session_id> --sort ts`
3. Fetch full transcript:
   - `memex session <session_id>`

## Indexing

- Incremental index:
  - `memex index`
- Embeddings are **off** by default.
- Enable during indexing:
  - `memex index --embeddings`
- Backfill embeddings for existing docs:
  - `memex embed`

## Config

Create `~/.memex/config.toml` (or `<root>/config.toml`):

```toml
embeddings = false
auto_index_on_search = true
model = "gemma"  # minilm, bge, nomic, gemma, potion
compute_units = "ane"  # macOS only: ane, gpu, cpu, all
scan_cache_ttl = 3600
```

`auto_index_on_search` refreshes index state before search and sessions listing.

## Output Parsing

Default output is JSONL (one JSON object per line).

Key fields:
- `doc_id`, `session_id`, `ts`, `project`, `role`, `source_path`
- `text`, `snippet`, `matches`, `score`

- `doc_id`: Unique record ID.
- `session_id`: Conversation thread ID.
- `ts`: ISO 8601 timestamp.
- `source`: Agent source name.
- `source_path`: Transcript path.
- `role`: `user`, `assistant`, `tool_use`, or `tool_result`.
- `text`: Content payload.
- `score`: Search relevance (float).
- `event_id`, `parent_session_id`, `conversation_kind`: Fork/tree metadata when available.

**Interpretation:**

- **Filtering:** Discard results below a relevance threshold (e.g., `score < 0.5`) unless specific.
- **Ordering:** Sort by `ts` for timeline reconstruction.
- **Grouping:** Aggregate by `session_id` to view conversation turns.
- **Forks:** Use `parent_session_id` to connect forked Opencode sessions back to the parent thread.

For downstream processing:
- Use `--json-array` for one JSON array.
- Use `--fields` to reduce payload size.
