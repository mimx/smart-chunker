"""Tests for PreChunkCompiler — rescue, idempotency, marker inference."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pre_chunk_compiler import PreChunkCompiler
from smart_chunker import SmartChunker, classify_for_chunking, CHUNKING_CONFIGS

SAMPLE_DOC = """@@IDENTITY:CROSSSYSTEM

# Identity Attribute Map

## Critical Gotchas

### Attribute A does not equal Attribute B
The A attribute does not match B in most cases.
Mapping requires an explicit join. Frequent source of confusion.

### Canonical username format
Usernames are stored lowercased without dots.

### Prefix inconsistency
The prefix sometimes uses variant-1 and sometimes variant-2. Normalise first.

### UUID is the only stable key
Only the global UUID survives renames. Everything else can change.

### Email is not unique over time
Reused emails break naive joins; always scope by active status.

## Join Keys
The primary join key is the UUID, with a secondary fallback when UUID is not yet provisioned.
"""

GOTCHA_HEADERS = [
    "Attribute A does not equal Attribute B",
    "Canonical username format",
    "Prefix inconsistency",
    "UUID is the only stable key",
    "Email is not unique over time",
]

def _kept_headers(content, source_path="knowledge/test.md"):
    doc_type = classify_for_chunking(content, source_path)
    chunker  = SmartChunker(CHUNKING_CONFIGS.get(doc_type, CHUNKING_CONFIGS["default"]))
    return {elem for c in chunker.chunk(content) for elem in (c.section_path or [])}

compiler = PreChunkCompiler()

def test_raw_chunker_drops_gotcha_sections():
    kept    = _kept_headers(SAMPLE_DOC)
    dropped = [h for h in GOTCHA_HEADERS if h not in kept]
    assert dropped, f"Expected some gotcha sections to be dropped without compiling; kept={kept}"

def test_compile_rescues_all_gotcha_sections():
    result  = compiler.compile(SAMPLE_DOC)
    kept    = _kept_headers(result["compiled_content"])
    missing = [h for h in GOTCHA_HEADERS if h not in kept]
    assert not missing, f"Compiler failed to rescue: {missing}"

def test_compile_report_counts():
    report = compiler.compile(SAMPLE_DOC)["report"]
    assert report["rescued_sections"] >= len(GOTCHA_HEADERS)
    assert report["after"]["dropped_count"] < report["before"]["dropped_count"]

def test_compile_is_idempotent():
    first  = compiler.compile(SAMPLE_DOC)
    second = compiler.compile(first["compiled_content"])
    assert second["report"]["transformations_count"] == 0

def test_compile_injects_marker_for_unmarked_doc():
    doc    = "# Runbook\n\n## Step\nDo the thing.\n"
    result = compiler.compile(doc, source_path="runbooks/deploy.md")
    assert result["report"]["document_marker"] is not None
    assert result["report"]["document_marker"] in result["compiled_content"]

def test_analyze_is_non_mutating():
    report = compiler.analyze(SAMPLE_DOC)
    assert report["dropped_count"] >= len(GOTCHA_HEADERS)
    assert "distribution" in report
