# Personal fork upstream sync — 2026-09-19

Merged upstream through `6044371` (v0.23.1), including the new readers,
checkpoint and lexical storage, stemming, sources, usage reporting, and inference
runtime. Retained resumable embedding jobs, supervised workers, confirmed-missing
file pruning, federated session listing, and memory section embedding reuse.

## Database preservation

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

## Validation

The migration regression covers changed IDs, duplicate text, UTF-8 truncation,
missing and removed inputs, incomplete export recovery, incompatible models and
dimensions, and corrupt/nonfinite vectors. A real BGE sample reused five of six
new records from four exported inputs and computed only the one new embedding.
Old/new CPU and automatic-provider BGE outputs matched above 0.999999999999 cosine
similarity on short, code, Unicode, and long inputs using the existing model cache.

The deployed model is BGE. Gemma with the new automatic/CoreML provider produced
NaNs in an upstream similarity test; this is not validated for deployment.
Native companion GUI applications and CUDA builds are outside this CLI rollout.
