# Personal fork upstream sync — 2026-09-19

Reviewed upstream through `6044371` (v0.23.1). This remains the personal v0.19
storage/parser baseline with selected backports, not a v0.23 installation.

## Included

- #177 (`6de38d0`): reuse memory lexical indexes and readers.
- #178 (`9284b4a`): bound context hydration, retaining legacy-index support.
- #186 (`8095e10`), usage portion: compact cached usage events and preallocate
  Codex usage deduplication. Keep this fork's activity API; omit the companion
  UI changes and unrelated analytics cleanup.
- #204 (`adba347`): canonicalize OAuth SQLite paths, including macOS temporary
  paths.

Parser versions, conversation/memory vector formats, model inputs, index
schemas, and Cargo dependencies are unchanged. The fork retains resumable
embedding backfill, worker supervision, durable ingestion, federated session
listing, and environment overrides for execution providers.

## Deferred upstream changes

The new conversation readers/parsers (#160–162), storage/ingestion series
(#166, #169–175), later source integrations, large usage rewrite (#194), and
ONNX/CUDA runtime update (#176) are not included. Companion app/release changes
are also outside this CLI update.

A blanket upgrade is unsafe for the existing database:

- Upstream writable index opening recreates projections missing
  `reader_metadata`; recovery clears vectors when it finds an empty lexical
  projection with populated ingest state.
- Upstream `vector_migration` globally rebuilds vectors on parser-version
  invalidation, even when most embedding inputs are identical.
- #175 switches Tantivy dictionaries from FST to SSTable. The current upstream
  binary rejects the old format; its reindex reset also removes vector and
  memory directories.

## Feasible migration without recomputing unchanged embeddings

The vector-generation format itself is compatible. The migration problem is
associating vectors with newly parsed records, whose document IDs may change.
A future migration can:

1. Pause writers and, using an old-format-compatible binary, stream existing
   user/assistant records and extract their vectors with `VectorIndex::embedding`.
2. Store vectors in a temporary SQLite database outside the directories that
   reindex resets. Key by model, dimensions, and SHA-256 of the exact embedding
   input: the first 8192 bytes of text, truncated at a UTF-8 boundary. Preserve
   the memory section cache separately.
3. Build the new lexical projection in a separate root. For each new eligible
   record, reuse a matching vector under its new document ID. Run inference
   only for unmatched inputs; do not match by document ID or approximate text.
4. Verify coverage, search, model identity, and restart recovery before switching
   the active root and removing the previous store/export.

At 280,000 vectors and 384 dimensions the raw f32 export is about 410 MiB,
plus SQLite/index overhead. Lexical rebuilding is still required, but a full
model pass is not inherently necessary. This route was assessed from source;
the exporter/importer is not implemented or validated by this update.
