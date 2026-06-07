"""Tests for SmartChunker — hierarchy, strong-signal, summary, quality score."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from smart_chunker import (
    SmartChunker, ChunkEnhancer, ChunkingConfig, CHUNKING_CONFIGS,
    INGESTION_VERSION, classify_for_chunking, BGE_MAX_TOKENS,
)


def _big_nested_doc():
    filler = " ".join(["word"] * 80)
    return (
        f"# Root Heading\n{filler}\n\n"
        f"## Alpha\n{filler}\n\n"
        f"### Alpha Deep\n{filler}\n\n"
        f"## Beta\n{filler}\n\n"
        f"### Beta Deep\n{filler}\n\n"
        f"#### Beta Deeper\n{filler}\n\n"
    )


def test_ers_marker_detection():
    assert "@@DOMAIN:SECTION" in SmartChunker()._detect_ers_markers("@@DOMAIN:SECTION some text")


def test_ers_marker_deduplication():
    markers = SmartChunker()._detect_ers_markers("@@A:B @@A:B @@C:D")
    assert markers == ["@@A:B", "@@C:D"]


def test_operation_detection():
    assert "add" in SmartChunker()._detect_operations("Add user to group")


def test_entity_detection():
    entities = SmartChunker()._detect_entities("A user joined the group")
    assert "user" in entities and "group" in entities


def test_api_method_detection():
    assert SmartChunker()._detect_api_method("PUT /api/v1/users/{id}") == "PUT"


def test_header_stack_full_ancestry():
    chunks = SmartChunker().chunk_document(_big_nested_doc(), title="Doc")
    deep = [c for c in chunks if c.parent_header == "Alpha Deep"]
    assert deep and deep[0].section_path == ["Doc", "Root Heading", "Alpha", "Alpha Deep"]


def test_header_stack_sibling_isolation():
    chunks = SmartChunker().chunk_document(_big_nested_doc(), title="Doc")
    beta_deep = [c for c in chunks if c.parent_header == "Beta Deep"]
    assert beta_deep and "Alpha" not in beta_deep[0].section_path


def test_strong_signal_ers_marker():
    assert SmartChunker()._has_strong_signal("@@DOMAIN:SECTION")


def test_strong_signal_error_code():
    assert SmartChunker()._has_strong_signal("AADSTS50011 means redirect mismatch")


def test_strong_signal_http_path():
    assert SmartChunker()._has_strong_signal("GET /api/v1/users")


def test_no_false_positive_on_plain_text():
    assert not SmartChunker()._has_strong_signal("The sky is blue today")


def test_atomic_fact_chunk_type():
    chunks = SmartChunker().chunk_document("# Errors\n\nAADSTS50011 means the reply URL does not match.", title="T")
    assert any(c.chunk_type == "atomic_fact" for c in chunks)


def test_header_only_divider_dropped():
    body = ("Authorization as a Service uses OPA to enforce policies. " * 12)
    doc  = "# Guide\n\n## @@API:SECTION — Concepts\n\n### Intro\n\n" + body
    chunks = SmartChunker().chunk_document(doc, title="Guide")
    for c in chunks:
        assert c.content.strip() != "## @@API:SECTION — Concepts"


def test_summary_chunk_present():
    chunks = SmartChunker().chunk_document(_big_nested_doc(), title="Doc")
    summaries = [c for c in chunks if c.chunk_type == "document_summary"]
    assert summaries and summaries[0].chunk_index == 0


def test_continuation_flags_on_split():
    big = "# Huge\n\n" + "\n\n".join(["Sentence. " * 60 for _ in range(10)])
    chunks = SmartChunker().chunk_document(big, title="Big")
    assert any(c.has_continuation for c in chunks) and any(c.is_continuation for c in chunks)


def test_classify_incident_by_path():
    assert classify_for_chunking("...", "incidents/2026-rca.md") == "incident"


def test_classify_api_reference_by_content():
    content = "\n".join([f"GET /api/v1/resource{i}\nReturns it." for i in range(10)])
    assert classify_for_chunking(content, "anything.md") == "api_reference"


def test_quality_score_range():
    chunks = SmartChunker().chunk_document(_big_nested_doc(), title="T")
    assert all(0 <= c.quality_score <= 100 for c in chunks)


def test_quality_score_higher_with_ers_markers():
    with_m    = SmartChunker().chunk_document("# S\n\n@@A:B marker text " + "word " * 80, title="T")
    without_m = SmartChunker().chunk_document("# S\n\n" + "word " * 80, title="T")
    s_with    = [c.quality_score for c in with_m    if c.chunk_type == "normal"]
    s_without = [c.quality_score for c in without_m if c.chunk_type == "normal"]
    if s_with and s_without:
        assert max(s_with) > max(s_without)


def test_count_tokens_real_without_tokenizer():
    assert SmartChunker()._count_tokens_real("hello world") is None


def test_ingestion_version():
    assert INGESTION_VERSION == "smart_chunker_v1.2"
