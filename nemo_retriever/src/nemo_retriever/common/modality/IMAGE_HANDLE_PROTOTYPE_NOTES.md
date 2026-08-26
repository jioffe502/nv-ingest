# Lossless image-handle prototype notes

Question: can a content-addressed image handle preserve exact bytes, fail
closed before embedding on missing/corrupt data, and remain live until a
terminal VDB receipt makes release safe?

The prototype models one page image and an in-memory blob store. It is not
performance evidence and must not be shipped as production code.

Initial verdict to confirm interactively:

- The handle needs version, URI, SHA-256, byte length, and media type.
- Rehydration owns strict length and digest verification. Missing or corrupt
  bytes stop embedding; there is no text-only fallback for a declared visual
  row.
- Rehydrated base64 is actor-local and is discarded immediately after model
  execution.
- An internally managed image object cannot be released until the terminal VDB
  write receipt commits. User-managed durable image stores require a different
  retention policy and must not be garbage-collected by ingest.
- A distributed production mode needs a shared fsspec-compatible URI. A local
  path is only safe for a single-node topology.

Run from `nemo_retriever`:

```bash
PYTHONPATH=src uv run --no-sync python -m nemo_retriever.common.modality.image_handle_tui_prototype
```

Delete the TUI and prototype state module after the contract is accepted or
absorbed into production types.
