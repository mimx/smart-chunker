# smart-chunker

Structure-aware document chunking for AI knowledge systems.

Generic text splitters break documents at token boundaries — splitting code blocks mid-line, orphaning section context, and discarding short-but-critical facts like API paths or error codes. `smart-chunker` solves this for Markdown-based knowledge bases.

## What it does

**`SmartChunker`** — splits a Markdown document into retrieval-optimised chunks that:

- Preserve full section ancestry (`H1 > H2 > H3`) in every chunk's `section_path`
- Keep high-signal short facts as `atomic_fact` chunks instead of discarding them
- Apply per-document-type token budgets (`api_reference`, `sop`, `concept`, `incident`, `json`, `default`)
- Extract semantic metadata per chunk (ERS markers, HTTP methods, CRUD operations, IAM entities)
- Produce a `document_summary` chunk (index 0) for broad queries on large documents
- Score every chunk with a signal-density `quality_score` (0–100) for retrieval boosting

**`PreChunkCompiler`** — a pre-processing pass that rescues sections SmartChunker would otherwise discard:

- Predicts which sections will be dropped (too small, no strong signal)
- Injects the nearest in-scope ERS routing marker to preserve them as `atomic_fact` chunks
- Returns a detailed audit report: sections rescued, transformations applied, remaining dropped

Together they form an ingestion pipeline where knowledge structured at authoring time survives through chunking and is available — with full context — at query time.

## Background

This library was extracted from a production RAG system for technical operations documentation. The design decisions are described in detail in:

- *[Stop Feeding Your Agents Prose — Structure Your Knowledge Like Code](https://mimx.github.io/posts/2026-06-07-stop-feeding-your-agents-prose/)* — the ERS authoring convention this chunker is designed to preserve
- *The Invisible Tax on Every AI Agent Query* — why generic chunking fails for technical domains and how SmartChunker addresses it (coming soon)

## Installation

No package yet — copy `smart_chunker.py` and optionally `pre_chunk_compiler.py` into your project.

**Dependencies:** pure Python standard library + `logging`. No external packages required for the chunker itself. `structlog` is optional (used in the original RAG API); the public version uses the standard `logging` module.

```bash
# run the tests
python -m pytest tests/ -v
```

## Quick start

```python
from smart_chunker import SmartChunker, CHUNKING_CONFIGS

# Default config
chunker = SmartChunker()

# Per-document-type config
chunker = SmartChunker(CHUNKING_CONFIGS["api_reference"])

chunks = chunker.chunk_document(markdown_text, title="My Document")

for chunk in chunks:
    print(chunk.chunk_type)      # normal | atomic_fact | document_summary
    print(chunk.section_path)    # ["Doc Title", "Section", "Subsection"]
    print(chunk.quality_score)   # 0-100
    print(chunk.metadata)        # ers_markers, operations, entities, api_method, api_endpoint
```

### With the Pre-Chunk Compiler

```python
from pre_chunk_compiler import PreChunkCompiler

compiler = PreChunkCompiler()

# Dry-run: see what would be dropped
report = compiler.analyze(markdown_text, source_path="docs/api.md")
print(f"Would drop {report['dropped_count']} of {report['section_count']} sections")

# Compile: rescue dropped sections
result   = compiler.compile(markdown_text, source_path="docs/api.md")
safe_md  = result["compiled_content"]   # pass this to SmartChunker
report   = result["report"]
print(f"Rescued {report['rescued_sections']} sections")

# Now chunk the safe content
chunks = SmartChunker().chunk_document(safe_md, title="My Doc")
```

## ERS markers

ERS (Extension Routing Structure) markers are `@@DOMAIN:SECTION` tags embedded in Markdown content. They act as routing anchors for retrieval systems — enabling domain-isolated search without scanning unrelated content.

```markdown
@@ORDERS:SCHEMA

## Orders Table

orders PK=ORDERKEY; status: 1=pending 2=confirmed 3=shipped 4=delivered 5=cancelled.
FK: orders.CUSTOMERKEY→customers.CUSTOMERKEY
```

SmartChunker detects these markers and stores them in `chunk.metadata["ers_markers"]`, enabling payload-filtered retrieval in vector stores like Qdrant, Weaviate, or Pinecone.

## Document types and token budgets

| Type | Target | Max | Min | Overlap | Use for |
|------|--------|-----|-----|---------|---------|
| `default` | 350 | 480 | 100 | 50 | Mixed documentation |
| `api_reference` | 180 | 360 | 40 | 10 | API endpoint docs |
| `json` | 130 | 260 | 30 | 0 | JSON schemas |
| `sop` | 220 | 440 | 80 | 30 | Runbooks, procedures |
| `concept` | 430 | 470 | 150 | 80 | Architecture overviews |
| `incident` | 270 | 430 | 80 | 40 | Postmortems, RCAs |

The document type is auto-detected from the source path and content via `classify_for_chunking()`, or you can pass a config directly.

## Chunk types

| Type | Meaning |
|------|---------|
| `normal` | Standard chunk within token budget |
| `atomic_fact` | Short but high-signal — API path, error code, schema key |
| `document_summary` | Auto-generated outline of section headers (index 0) |

## Quality score

Every chunk gets a `quality_score` (0–100) computed from signal density:

| Signal | Points |
|--------|--------|
| Baseline | +50 |
| ERS markers present | +15 |
| IAM entity keywords | +2 each, capped +10 |
| HTTP method + path | +10 |
| CRUD operation verbs | +2 each, capped +10 |
| `atomic_fact` type | +5 |
| `document_summary` type | +5 |
| Token count < 50 | -10 |

Use this score for retrieval-time boosting — it breaks ties between chunks with similar cosine similarity in favour of operationally specific content.

## License

MIT
