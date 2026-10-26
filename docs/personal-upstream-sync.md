# Personal fork upstream sync

## v0.31.1 (2026-10-26)

Merged the released upstream tag through `a205ca7`: read-time session freshness,
literal fallback for malformed search syntax, safe resume quoting, post-publication
cleanup fixes, Goose support, and client and usage corrections. The subsequent
release-authoring skill is outside this runtime update.

Retained vector-preserving parser updates, confirmed-missing pruning, durable
directories, environment provider precedence, and the CPU default. Goose virtual
session paths are excluded from filesystem-based pruning. Published-snapshot reads
remain available during ordinary ingest, but explicit `index_freshness = "strict"`
must acquire the lease rather than use that fallback. The existing setting
`auto_index_on_search = false` continues to disable read-time refreshes.

### Embedding preservation

Claude/Codex parser versions, vector format, lexical schema, and embedding runtime
dependencies are unchanged. Upstream's parser-triggered whole-vector reset remains
removed. No index rebuild, model change, or vector import is needed.

An isolated copy of the Mac index contained 10,378 BGE vectors (384 dimensions).
Normal lexical ingestion with v0.31.1 preserved all 10,366 surviving vectors bit
for bit. An independent v0.28.0 control retired exactly the same 12 old record IDs
as live transcripts changed; there was no upgrade-specific invalidation. All 3,067
entries in the memory-section embedding cache retained their exact bytes. Semantic
search and pasted-URL lexical search passed on the staged index.

The explicit real-BGE parser-change regression passed: exactly one replacement
record was embedded, the unaffected vector stayed identical, and repeat ingestion
performed zero inference. The ordinary regression suite also covers lexical-only
parser changes and refusal to discard vectors during empty-index recovery.

Validation: formatting, strict Clippy across all targets, and all 1,427 default
tests passed (three ignored); the BGE regression above passed separately.

## PR #250 and v0.28.0 (2026-10-14)

Merged upstream through `c59f46f`, including the final multi-profile session and
usage accounting changes. Claude workflow copies now contribute their unique
history to one session, and resuming a default profile clears an inherited
`CLAUDE_CONFIG_DIR`. Retained personal pruning, published-snapshot reads, durable
directories, vector-preserving parser updates, and the macOS CPU default.

The upstream Claude index parser version remains 8; its usage parser moves to 6.
No lexical schema or vector format migration is needed for this update.

## Claude profile roots (2026-10-12)

`yaugenst-flex/fix-claude-profile-roots`, based on `upstream/main`, carries the
upstreamable series: one copy per duplicated workflow agent transcript, one usage
count per response without a request ID, the `memex sessions` argument struct,
`claude_paths` passed explicitly to ingest, usage and the resume guard, the
`{claude_env}` resume placeholder, `memex sessions --profile`, retiring transcripts
that moved between Claude roots, and in-place `vector_transfer` imports. This branch
carries the same commits with two fork-only differences:

- The moved-transcript rule lives in `src/ingest/prune.rs` instead of upstream's
  inline pruning loop, so its commit differs from the upstream one.
- `memex prune` reads the configured Claude roots (its own commit).

### Embedding preservation

The live root moved from `~/.claude/projects` to per-profile roots without
recomputing vectors: export from the stopped root, a lexical `index --no-embeddings`
on a reflink copy, then a top-up `vector_transfer import`. All 215,359 vectors were
kept or restored bit for bit; the 8,117 inputs never embedded before were left to
the daemon's backfill. See "Keep vectors when transcripts move" in
`docs/vector-migration.md`.

## v0.27.1 (2026-10-05)

Merged upstream through `61194bf` (v0.27.1): conversation relevance ranking,
one-pass semantic filters, incompatible-index daemon recovery, expanded stats,
remote embeddings, and Hermes/KiloCode sources. Retained personal pruning,
published-snapshot reads, durable directories, environment provider precedence,
and the macOS CPU default. The expanded stats report also retains segment count
and index storage size.

### Embedding preservation

This upgrade changes neither the lexical schema nor the Claude/Codex parser
versions. Existing BGE vectors retain their `bge` identity and 384 dimensions;
local embeddings remain local unless remote mode is explicitly configured.
The memory-section cache and resumable backfill continue to reuse saved vectors.

Keep upstream's parser-triggered `vector_migration` / `reset_vector_store` path
removed. A parser change replaces only the affected records and preserves all
unaffected vectors; lexical-only ingestion defers replacement embeddings. The
BGE regression below exercises enabled inference, asserts exactly one replacement
embedding, checks the unaffected vector is unchanged, and verifies a repeat ingest
performs zero inference:

```sh
MEMEX_EXECUTION_PROVIDER=cpu cargo test --lib \
  parser_change_with_bge_embeds_only_replaced_records -- --ignored
```

### Validation and deployment boundary

- Formatting and Clippy (`--all-targets -- -D warnings`) pass.
- Library tests: 1,216 passed, three ignored by default; the additional BGE
  preservation test above passed when explicitly enabled.
- All 97 integration tests pass, including daemon recovery and retrieval-quality
  regression tests. The daemon deletion test now waits for both lexical and
  vector publication; lexical publication alone can precede vector publication.
- v0.27.1 stats and semantic search read an isolated APFS clone of the existing
  2,524,641-record index with all 283,079 BGE vectors. SHA-256 checks of all saved
  vector files match before/after search, both in the clone and the live root.
- The installed release binary, services, live index, and user configuration
  were not changed. Deployment and native GUI/CUDA validation are separate.

## v0.24.0 (2026-09-23)

Merged upstream through `649355e` (v0.24.0). Upstream now carries our PRs #205
to #210: watcher access-event filtering, federated session listings, atomic
analytics rebuilds, explicit lexical migrations that preserve vectors, and
resumable supervised embedding backfills. Conflicts were resolved to the
upstream versions of those changes. This branch still adds `memex prune` and
confirmed-missing pruning during ingest, published-snapshot reads while
auto-index is busy, index size in `memex stats`, durable store directory
creation, and `MEMEX_EXECUTION_PROVIDER` precedence over the config file.

Upstream still clears the whole vector store when any parser version changes
(`vector_migration` and `reset_vector_store` in `src/ingest`). This branch keeps
that path removed so parser changes re-embed only the records they replace.
Future merges must drop it again.

Upstream resolves `execution_provider = "auto"` to CoreML on macOS. On
2026-09-28, a BGE backfill on CoreML with `compute_units = "ane"` grew one embed
worker past a 60 GB footprint on a 24 GB laptop and never finished; the same
backfill on CPU peaked near 4.5 GB and completed in about four minutes. This
branch resolves `auto` to CPU everywhere (`ExecutionProviderChoice::effective`),
so CoreML is explicit opt-in. Future merges must keep that default.

v0.24.0 changed no lexical schema or parser versions. The usage cache drops its
`usage_file_cache` table once on first open.

## v0.23.1 (2026-09-19)

Merged upstream through `6044371` (v0.23.1), including the new readers,
checkpoint and lexical storage, stemming, sources, usage reporting, and inference
runtime. Retained resumable embedding jobs, supervised workers, confirmed-missing
file pruning, federated session listing, and memory section embedding reuse.

### Database preservation

Writable opens and TUI startup reject stale lexical schemas without replacing
files. Parser-version changes no longer clear the entire vector store; unchanged
record IDs retain vectors, changed records enter resumable backfill. Recovery
refuses to erase vectors when a populated checkpoint has an empty lexical index.
Explicit `index rebuild` remains destructive and must not be used as a migration
against the active root.

The offline `vector_transfer` example transfers vectors across storage versions:

1. Stop the daemon and embedding workers. Build the exporter against the old
   storage code; export to a temporary SQLite file outside both database roots.
2. Build a fresh lexical root with the new binary and `index rebuild
   --no-embeddings --root STAGE`. Keep the existing configuration and model.
3. Import with the new helper. It maps SHA-256 hashes of exact embedding input
   (first 8192 bytes, truncated at a UTF-8 boundary) to new record IDs. Model,
   dimensions, format, completeness, finite values, and exact saved f32 values
   are checked. Investigate unmatched exported hashes before switching.
4. Preserve the memory documents, vectors, and section cache while writers are
   stopped. Run resumable embedding backfill only for genuinely missing inputs.
5. Verify search, memory, counts, and service restart before switching roots;
   remove the temporary old root and export after successful validation.

Export checkpoints can resume against the same stopped source snapshot. Import
publishes only after all matched vectors pass bitwise checks, then verifies the
reopened generation. This is an operator-driven migration, not automatic startup
behavior. The original root remains untouched until the final directory switch.

### Validation

The migration regression covers changed IDs, duplicate text, UTF-8 truncation,
missing and removed inputs, incomplete export recovery, incompatible models and
dimensions, and corrupt/nonfinite vectors. A real BGE sample reused five of six
new records from four exported inputs and computed only the one new embedding.
Old/new CPU and automatic-provider BGE outputs matched above 0.999999999999 cosine
similarity on short, code, Unicode, and long inputs using the existing model cache.

The deployed model is BGE. Gemma with the new automatic/CoreML provider produced
NaNs in an upstream similarity test; this is not validated for deployment.
Native companion GUI applications and CUDA builds are outside this CLI rollout.
