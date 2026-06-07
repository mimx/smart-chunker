"""
PreChunkCompiler — Deterministic pre-processing for SmartChunker.

Problem it solves:
  SmartChunker drops any section below min_tokens that lacks a "strong signal".
  Short sections (e.g. a 20-token gotcha note) contain critical facts but are
  silently discarded. The compiler rescues them by injecting the nearest in-scope
  ERS marker, flipping _has_strong_signal() to True without changing any threshold.

Strategy:
  1. Parse the document with the same parser SmartChunker uses.
  2. Predict each section's fate under the selected ChunkingConfig.
  3. For sections predicted dropped (but with body text), inject the nearest
     ERS marker — preserving them as atomic_fact chunks.
  4. Guarantee the document carries at least one routing ERS marker.

The compiler is a pure input transformer: SmartChunker is unchanged.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging

from smart_chunker import (
    CHUNKING_CONFIGS,
    SmartChunker,
    Section,
    classify_for_chunking,
)

logger = logging.getLogger(__name__)

COMPILER_VERSION = "prechunk_compiler_v1"

_MARKER_RE = re.compile(r"@@[A-Z0-9_]+:[A-Z0-9_:]+")

# Domain → default marker inference for documents with no ERS marker.
# Extend this table for your own domains.
_DOMAIN_MARKER_HINTS: List[Tuple[Tuple[str, ...], str]] = [
    (("incident", "rca", "postmortem"),           "@@INCIDENT:RUNBOOK"),
    (("security", "secops"),                       "@@SECURITY:AUDIT"),
    (("identity", "iam", "crosssystem"),           "@@IDENTITY:CROSSSYSTEM"),
    (("api-reference", "openapi"),                 "@@API:REFERENCE"),
    (("runbook", "sop", "procedure", "playbook"),  "@@OPS:RUNBOOK"),
]


@dataclass
class SectionFate:
    header: Optional[str]
    level: int
    section_path: List[str]
    tokens: int
    has_signal: bool
    fate: str  # kept_normal | kept_atomic_fact | split | dropped_tiny | dropped_header_only | dropped_empty

    def to_dict(self) -> Dict:
        return {"header": self.header, "level": self.level, "section_path": self.section_path,
                "tokens": self.tokens, "has_signal": self.has_signal, "fate": self.fate}


@dataclass
class Transformation:
    action: str          # inject_marker | add_document_marker
    section: Optional[str]
    marker: str
    reason: str

    def to_dict(self) -> Dict:
        return {"action": self.action, "section": self.section,
                "marker": self.marker, "reason": self.reason}


_DROPPED = {"dropped_tiny", "dropped_header_only", "dropped_empty"}


class PreChunkCompiler:
    """
    Reshapes Markdown documents so SmartChunker preserves every meaningful section.

    Usage::

        from pre_chunk_compiler import PreChunkCompiler

        compiler = PreChunkCompiler()

        # Dry-run: see what would be dropped
        report = compiler.analyze(markdown_text, source_path="docs/api.md")

        # Compile: rescue dropped sections and get the safe content
        result  = compiler.compile(markdown_text, source_path="docs/api.md")
        safe_md = result["compiled_content"]
        report  = result["report"]   # rescued_sections, transformations, etc.
    """

    @staticmethod
    def _chunker_for(content: str, source_path: str) -> Tuple[SmartChunker, str]:
        doc_type = classify_for_chunking(content, source_path)
        config   = CHUNKING_CONFIGS.get(doc_type, CHUNKING_CONFIGS["default"])
        return SmartChunker(config=config), doc_type

    @staticmethod
    def _markers_in(text: str) -> List[str]:
        seen: List[str] = []
        for m in _MARKER_RE.findall(text or ""):
            if m not in seen:
                seen.append(m)
        return seen

    @classmethod
    def _infer_document_marker(cls, content: str, source_path: str) -> Optional[str]:
        haystack = f"{source_path}\n{content[:2000]}".lower()
        for needles, marker in _DOMAIN_MARKER_HINTS:
            if any(n in haystack for n in needles):
                return marker
        return None

    def _predict_fate(self, chunker: SmartChunker, section: Section) -> SectionFate:
        body       = chunker._section_body(section).strip()
        tokens     = chunker._estimate_tokens(section.content)
        has_signal = chunker._has_strong_signal(section.content)
        path       = section.full_path or ([section.header] if section.header else [])

        if section.header and not body:            fate = "dropped_header_only"
        elif tokens > chunker.max_tokens:          fate = "split"
        elif tokens >= chunker.min_tokens:         fate = "kept_normal"
        elif has_signal:                           fate = "kept_atomic_fact"
        elif section.content.strip():              fate = "dropped_tiny"
        else:                                      fate = "dropped_empty"

        return SectionFate(header=section.header, level=section.header_level,
                           section_path=path, tokens=tokens,
                           has_signal=has_signal, fate=fate)

    @staticmethod
    def _distribution(fates: List[SectionFate]) -> Dict[str, int]:
        dist: Dict[str, int] = {}
        for f in fates:
            dist[f.fate] = dist.get(f.fate, 0) + 1
        return dist

    @staticmethod
    def _inject_marker(section_content: str, marker: str) -> str:
        lines = section_content.split("\n")
        if lines and re.match(r"^#{1,6}\s+", lines[0]):
            rest = lines[1:]
            insert = [lines[0], "", marker]
            if rest and rest[0].strip() == "":
                rest = rest[1:]
            return "\n".join(insert + ([""] + rest if rest else []))
        return f"{marker}\n\n{section_content}"

    def analyze(self, content: str, source_path: str = "") -> Dict:
        """Dry-run: report fate of every section without modifying content."""
        chunker, doc_type = self._chunker_for(content, source_path)
        sections = chunker._parse_sections(content, None)
        fates    = [self._predict_fate(chunker, s) for s in sections]
        kept     = [f for f in fates if f.fate not in _DROPPED]
        dropped  = [f for f in fates if f.fate in _DROPPED]
        return {
            "doc_type":      doc_type,
            "config":        {"min_tokens": chunker.min_tokens, "max_tokens": chunker.max_tokens,
                              "target_tokens": chunker.target_tokens},
            "ers_markers":   self._markers_in(content),
            "section_count": len(sections),
            "kept_count":    len(kept),
            "dropped_count": len(dropped),
            "distribution":  self._distribution(fates),
            "sections":      [f.to_dict() for f in fates],
        }

    def compile(self, content: str, source_path: str = "", default_marker: Optional[str] = None) -> Dict:
        """
        Return chunker-safe content plus an audit report.

        The report includes:
          - rescued_sections : how many dropped_tiny sections were rescued
          - transformations  : list of injections applied
          - remaining_dropped: sections still dropped after compile (truly empty)
        """
        chunker, doc_type = self._chunker_for(content, source_path)
        sections    = chunker._parse_sections(content, None)
        before_fates = [self._predict_fate(chunker, s) for s in sections]

        doc_markers = self._markers_in(content)
        doc_marker  = (doc_markers[0] if doc_markers
                       else (default_marker or self._infer_document_marker(content, source_path)))

        transformations: List[Transformation] = []
        out_parts: List[str] = []
        current_marker = doc_marker

        if doc_marker and not doc_markers:
            out_parts.append(doc_marker)
            transformations.append(Transformation(
                action="add_document_marker", section=None, marker=doc_marker,
                reason="document carried no ERS marker; added for routing"))

        for section, fate in zip(sections, before_fates):
            sec_markers = self._markers_in(section.content)
            if sec_markers:
                current_marker = sec_markers[0]

            new_content = section.content
            if fate.fate == "dropped_tiny":
                marker = current_marker or doc_marker
                if marker and marker not in section.content:
                    new_content = self._inject_marker(section.content, marker)
                    transformations.append(Transformation(
                        action="inject_marker",
                        section=" > ".join(fate.section_path) or fate.header,
                        marker=marker,
                        reason=f"~{fate.tokens} tok < min_tokens={chunker.min_tokens}; would be silently dropped"))

            out_parts.append(new_content)

        compiled = "\n".join(out_parts)

        after_chunker, _ = self._chunker_for(compiled, source_path)
        after_sections   = after_chunker._parse_sections(compiled, None)
        after_fates      = [self._predict_fate(after_chunker, s) for s in after_sections]

        before_dropped = sum(1 for f in before_fates if f.fate in _DROPPED)
        after_dropped  = sum(1 for f in after_fates  if f.fate in _DROPPED)

        return {
            "compiled_content": compiled,
            "report": {
                "compiler_version":  COMPILER_VERSION,
                "doc_type":          doc_type,
                "document_marker":   doc_marker,
                "ers_markers":       self._markers_in(compiled),
                "transformations":   [t.to_dict() for t in transformations],
                "transformations_count": len(transformations),
                "before":  {"section_count": len(sections),      "dropped_count": before_dropped, "distribution": self._distribution(before_fates)},
                "after":   {"section_count": len(after_sections), "dropped_count": after_dropped,  "distribution": self._distribution(after_fates), "sections": [f.to_dict() for f in after_fates]},
                "rescued_sections":  before_dropped - after_dropped,
                "remaining_dropped": [f.to_dict() for f in after_fates if f.fate in _DROPPED],
            }
        }
