"""
SmartChunker — Structure-aware document chunking for AI knowledge systems.

Splits Markdown documents into retrieval-optimised chunks while preserving:
  - Full section hierarchy (H1→H6 ancestry in every chunk)
  - High-signal short facts (API paths, error codes, schema keys) as atomic_fact chunks
  - Per-document-type token budgets (api_reference, sop, concept, incident, json, default)
  - Semantic metadata per chunk (ERS markers, HTTP methods, CRUD operations, entities)
  - Document summary chunk for broad queries

Version: smart_chunker_v1.2
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

INGESTION_VERSION = "smart_chunker_v1.2"
BGE_MAX_TOKENS = 512  # hard limit for BAAI/bge-small-en-v1.5


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ChunkingConfig:
    """
    Chunking parameters for a specific document type.

    Split decisions use the fast words*1.3 estimator (target/max/min/overlap).
    When a real tokenizer is available the chunker also validates that no chunk
    exceeds BGE_MAX_TOKENS (512) and stores token_count_real in the payload.
    """
    target_tokens: int = 350
    max_tokens: int = 480
    min_tokens: int = 100
    overlap_tokens: int = 50
    doc_type: str = "default"


CHUNKING_CONFIGS: Dict[str, ChunkingConfig] = {
    "default":       ChunkingConfig(),
    "api_reference": ChunkingConfig(target_tokens=180, max_tokens=360, min_tokens=40,  overlap_tokens=10,  doc_type="api_reference"),
    "json":          ChunkingConfig(target_tokens=130, max_tokens=260, min_tokens=30,  overlap_tokens=0,   doc_type="json"),
    "sop":           ChunkingConfig(target_tokens=220, max_tokens=440, min_tokens=80,  overlap_tokens=30,  doc_type="sop"),
    "concept":       ChunkingConfig(target_tokens=430, max_tokens=470, min_tokens=150, overlap_tokens=80,  doc_type="concept"),
    "incident":      ChunkingConfig(target_tokens=270, max_tokens=430, min_tokens=80,  overlap_tokens=40,  doc_type="incident"),
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Section:
    """Intermediate representation of one logical section in a document."""
    header: Optional[str]
    header_level: int
    content: str
    start_line: int
    end_line: int
    full_path: List[str] = field(default_factory=list)
    subsections: List['Section'] = field(default_factory=list)


@dataclass
class Chunk:
    """A chunk ready for embedding and vector-store indexing."""
    content: str
    chunk_index: int
    token_count: int
    metadata: Dict = field(default_factory=dict)

    parent_header: Optional[str] = None
    section_path: List[str] = field(default_factory=list)
    section_depth: int = 0
    has_continuation: bool = False
    is_continuation: bool = False
    chunk_type: str = "normal"   # normal | atomic_fact | document_summary

    token_count_real: Optional[int] = None
    quality_score: int = 50

    chunk_metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "content":           self.content,
            "chunk_index":       self.chunk_index,
            "token_count":       self.token_count,
            "token_count_real":  self.token_count_real,
            "quality_score":     self.quality_score,
            "parent_header":     self.parent_header,
            "section_path":      self.section_path,
            "section_depth":     self.section_depth,
            "has_continuation":  self.has_continuation,
            "is_continuation":   self.is_continuation,
            "chunk_type":        self.chunk_type,
            "metadata":          self.metadata,
            "chunk_metadata":    self.chunk_metadata,
        }


# =============================================================================
# SMARTCHUNKER
# =============================================================================

class SmartChunker:
    """
    Structure-aware document chunker.

    Usage::

        from smart_chunker import SmartChunker, CHUNKING_CONFIGS

        chunker = SmartChunker()                             # default config
        chunker = SmartChunker(CHUNKING_CONFIGS["sop"])      # SOP config
        chunks  = chunker.chunk_document(markdown_text, title="My Doc")

    Each returned Chunk carries:
      - section_path  : full ancestor breadcrumb [doc_title, h1, h2, ...]
      - chunk_type    : "normal" | "atomic_fact" | "document_summary"
      - quality_score : 0-100 signal-density heuristic
      - metadata      : ERS markers, HTTP methods, CRUD ops, IAM entities
    """

    def __init__(self, config: Optional[ChunkingConfig] = None, tokenizer=None, **legacy_kwargs):
        if config is not None:
            self.config = config
        elif legacy_kwargs:
            self.config = ChunkingConfig(
                target_tokens=legacy_kwargs.get("target_tokens", 350),
                max_tokens=legacy_kwargs.get("max_tokens", 480),
                min_tokens=legacy_kwargs.get("min_tokens", 100),
                overlap_tokens=legacy_kwargs.get("overlap_tokens", 50),
            )
        else:
            self.config = ChunkingConfig()

        self._tokenizer = tokenizer
        self.target_tokens = self.config.target_tokens
        self.max_tokens    = self.config.max_tokens
        self.min_tokens    = self.config.min_tokens
        self.overlap_tokens = self.config.overlap_tokens

    # ── semantic detectors ───────────────────────────────────────────────────

    def _detect_ers_markers(self, content: str) -> List[str]:
        """Detect ERS markers (@@DOMAIN:SECTION) — deduplicated."""
        seen: set = set()
        result: List[str] = []
        for domain, section in re.findall(r'@@([A-Z0-9_]+):([A-Z0-9_]+)', content):
            marker = f"@@{domain}:{section}"
            if marker not in seen:
                seen.add(marker)
                result.append(marker)
        return result

    def _detect_operations(self, content: str) -> List[str]:
        ops = {
            "add":    r'\b(add|adds|adding|assign|assigns|grant|grants|attach|attaches)\b',
            "remove": r'\b(remove|removes|removing|delete|deletes|deleting|revoke|revokes|detach|detaches)\b',
            "list":   r'\b(list|lists|listing|get all|fetch all|enumerate|enumerates)\b',
            "get":    r'\b(get|gets|retrieve|retrieves|fetch|fetches|find|finds|search|searches)\b',
            "update": r'\b(update|updates|updating|modify|modifies|change|changes|edit|edits)\b',
            "create": r'\b(create|creates|creating|new|register|registers)\b',
        }
        cl = content.lower()
        return [op for op, pat in ops.items() if re.search(pat, cl)]

    def _detect_entities(self, content: str) -> List[str]:
        entities = {
            "user":        r'\b(user|users|member|members|person|account|accounts)\b',
            "group":       r'\b(group|groups|team|teams|membership|memberships)\b',
            "app":         r'\b(app|apps|application|applications)\b',
            "role":        r'\b(role|roles|permission|permissions)\b',
            "entitlement": r'\b(entitlement|entitlements|access right|access rights)\b',
            "policy":      r'\b(policy|policies|rule|rules)\b',
            "device":      r'\b(device|devices|endpoint|endpoints)\b',
            "factor":      r'\b(factor|factors|mfa|2fa|authenticator|authenticators)\b',
        }
        cl = content.lower()
        return [ent for ent, pat in entities.items() if re.search(pat, cl)]

    def _detect_api_method(self, content: str) -> Optional[str]:
        m = re.search(r'\b(GET|POST|PUT|DELETE|PATCH)\s+(?:\{[^}]+\})?/[A-Za-z0-9_./{}:-]+', content, re.IGNORECASE)
        return m.group(1).upper() if m else None

    def _detect_api_endpoint(self, content: str) -> Optional[str]:
        m = re.search(r'\b(?:GET|POST|PUT|DELETE|PATCH)\s+((?:\{[^}]+\})?/[A-Za-z0-9_./{}:-]+)', content, re.IGNORECASE)
        if m:
            return m.group(1)
        m2 = re.search(r'((?:\{[^}]+\})?/api/[A-Za-z0-9_./{}:-]+)', content)
        return m2.group(1) if m2 else None

    def _has_strong_signal(self, content: str) -> bool:
        """Return True if content contains high-value technical signals worth preserving."""
        patterns = [
            r'@@[A-Z0-9_]+:[A-Z0-9_]+',
            r'\b(GET|POST|PUT|DELETE|PATCH)\s+/[A-Za-z0-9_/{}]',
            r'\b[A-Z]{2,8}\d{4,}\b',
            r'\b(oauth2|saml|oidc|jwt|bearer|access_token)\b',
        ]
        for pat in patterns[:3]:
            if re.search(pat, content):
                return True
        if re.search(patterns[3], content.lower()):
            return True
        return False

    def _semantic_metadata(self, content: str, metadata: Dict) -> Dict:
        m = metadata.copy()
        m.update({
            "ers_markers":  self._detect_ers_markers(content),
            "operations":   self._detect_operations(content),
            "entities":     self._detect_entities(content),
            "api_method":   self._detect_api_method(content),
            "api_endpoint": self._detect_api_endpoint(content),
        })
        return m

    # ── public API ───────────────────────────────────────────────────────────

    def chunk_document(self, content: str, title: Optional[str] = None, metadata: Optional[Dict] = None) -> List[Chunk]:
        """
        Split a Markdown document into structured, retrieval-optimised chunks.

        Returns a list starting with an optional document_summary chunk (index 0)
        when the document has 3 or more distinct sections.
        """
        metadata = metadata or {}
        chunks: List[Chunk] = []

        for section in self._parse_sections(content, title):
            chunks.extend(self._process_section(section, len(chunks), metadata))

        if not chunks and content.strip():
            chunks = self._chunk_plain_text(content, title, metadata)

        for i, c in enumerate(chunks):
            if i < len(chunks) - 1 and chunks[i + 1].section_path == c.section_path:
                c.has_continuation = True
                chunks[i + 1].is_continuation = True

        summary = self.create_summary_chunk(chunks, title or "")
        if summary is not None:
            summary.chunk_index = 0
            for c in chunks:
                c.chunk_index += 1
            chunks.insert(0, summary)

        for c in chunks:
            c.token_count_real = self._count_tokens_real(c.content)
            c.quality_score    = self._compute_chunk_quality(c)

        return chunks

    def chunk(self, content: str, title: Optional[str] = None, metadata: Optional[Dict] = None) -> List[Chunk]:
        """Alias for chunk_document."""
        return self.chunk_document(content, title=title, metadata=metadata)

    # ── section parsing ──────────────────────────────────────────────────────

    def _parse_sections(self, content: str, doc_title: Optional[str] = None) -> List[Section]:
        sections: List[Section] = []
        lines = content.split('\n')
        header_stack: List[Tuple[int, str]] = []
        if doc_title:
            header_stack.append((0, doc_title))

        current_section: Optional[Section] = None
        current_lines: List[str] = []
        current_start = 0

        def flush(end_line: int) -> None:
            nonlocal current_section, current_lines, current_start
            if not current_lines and current_section is None:
                return
            joined = '\n'.join(current_lines)
            if current_section is None:
                sections.append(Section(header=None, header_level=0, content=joined,
                                        start_line=current_start, end_line=end_line,
                                        full_path=[t for _, t in header_stack]))
            else:
                current_section.content = joined
                current_section.end_line = end_line
                sections.append(current_section)
            current_section = None
            current_lines = []

        for i, line in enumerate(lines):
            hm = re.match(r'^(#{1,6})\s+(.+)$', line)
            if hm:
                flush(i - 1)
                level = len(hm.group(1))
                text  = hm.group(2).strip()
                while header_stack and header_stack[-1][0] >= level:
                    header_stack.pop()
                header_stack.append((level, text))
                current_section = Section(header=text, header_level=level,
                                          content="", start_line=i, end_line=i,
                                          full_path=[t for _, t in header_stack])
                current_lines = [line]
                current_start = i
            else:
                current_lines.append(line)

        flush(len(lines) - 1)
        return sections

    # ── section → chunks ─────────────────────────────────────────────────────

    @staticmethod
    def _section_body(section: Section) -> str:
        if not section.header:
            return section.content
        lines = section.content.split('\n')
        if lines and re.match(r'^#{1,6}\s+', lines[0]):
            return '\n'.join(lines[1:])
        return section.content

    def _process_section(self, section: Section, start_index: int, base_metadata: Dict) -> List[Chunk]:
        section_path  = section.full_path or ([section.header] if section.header else [])
        section_depth = len(section_path)

        if section.header and not self._section_body(section).strip():
            return []

        semantic  = self._semantic_metadata(section.content, base_metadata)
        tokens    = self._estimate_tokens(section.content)

        if tokens <= self.max_tokens:
            if tokens < self.min_tokens and not self._has_strong_signal(section.content):
                return []
            ctype = "atomic_fact" if tokens < self.min_tokens else "normal"
            return [Chunk(content=section.content, chunk_index=start_index,
                          token_count=tokens, parent_header=section.header,
                          section_path=section_path, section_depth=section_depth,
                          chunk_type=ctype, metadata=semantic.copy())]
        return self._split_section(section, section_path, section_depth, start_index, semantic)

    def _split_section(self, section, section_path, section_depth, start_index, metadata):
        chunks: List[Chunk] = []
        paragraphs = self._split_paragraphs(section.content)
        current_parts: List[str] = []
        current_tokens = 0

        if section.header:
            h = f"# {section.header}\n\n"
            current_parts.append(h)
            current_tokens = self._estimate_tokens(h)

        def flush(parts):
            text = '\n\n'.join(parts)
            tc   = self._estimate_tokens(text)
            if tc >= self.min_tokens or self._has_strong_signal(text):
                ctype = "atomic_fact" if tc < self.min_tokens else "normal"
                chunks.append(Chunk(content=text, chunk_index=start_index + len(chunks),
                                    token_count=tc, parent_header=section.header,
                                    section_path=section_path, section_depth=section_depth,
                                    chunk_type=ctype, metadata=metadata.copy(),
                                    chunk_metadata=self._semantic_metadata(text, {})))

        for para in paragraphs:
            pt = self._estimate_tokens(para)
            if current_tokens + pt <= self.max_tokens:
                current_parts.append(para); current_tokens += pt
            else:
                if current_tokens >= self.min_tokens or (current_parts and self._has_strong_signal('\n\n'.join(current_parts))):
                    flush(current_parts)
                    overlap = self._get_overlap(current_parts)
                    current_parts = [overlap] if overlap else []
                    current_tokens = self._estimate_tokens(overlap) if overlap else 0
                if pt > self.max_tokens:
                    chunks.extend(self._split_by_sentences(para, section.header, section_path, section_depth, start_index + len(chunks), metadata))
                    current_parts = []; current_tokens = 0
                else:
                    current_parts = [para]; current_tokens = pt

        if current_parts:
            flush(current_parts)
        return chunks

    def _split_by_sentences(self, text, parent_header, section_path, section_depth, start_index, metadata):
        chunks: List[Chunk] = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        current: List[str] = []; current_tokens = 0

        def flush_s(parts):
            joined = ' '.join(parts)
            tc = self._estimate_tokens(joined)
            if tc >= self.min_tokens or self._has_strong_signal(joined):
                ctype = "atomic_fact" if tc < self.min_tokens else "normal"
                chunks.append(Chunk(content=joined, chunk_index=start_index + len(chunks),
                                    token_count=tc, parent_header=parent_header,
                                    section_path=section_path, section_depth=section_depth,
                                    chunk_type=ctype, metadata=metadata.copy()))

        for sent in sentences:
            st = self._estimate_tokens(sent)
            if current_tokens + st <= self.max_tokens:
                current.append(sent); current_tokens += st
            else:
                if current: flush_s(current)
                if st > self.max_tokens:
                    chunks.extend(self._split_by_words(sent, parent_header, section_path, section_depth, start_index + len(chunks), metadata))
                    current = []; current_tokens = 0
                else:
                    current = [sent]; current_tokens = st
        if current: flush_s(current)
        return chunks

    def _split_by_words(self, text, parent_header, section_path, section_depth, start_index, metadata):
        chunks: List[Chunk] = []; words = text.split(); current: List[str] = []; ct = 0
        for word in words:
            wt = self._estimate_tokens(word)
            if ct + wt <= self.max_tokens:
                current.append(word); ct += wt
            else:
                if current:
                    joined = ' '.join(current)
                    chunks.append(Chunk(content=joined, chunk_index=start_index + len(chunks),
                                        token_count=self._estimate_tokens(joined), parent_header=parent_header,
                                        section_path=section_path, section_depth=section_depth,
                                        chunk_type="normal", metadata=metadata.copy()))
                current = [word]; ct = wt
        if current:
            joined = ' '.join(current)
            chunks.append(Chunk(content=joined, chunk_index=start_index + len(chunks),
                                token_count=self._estimate_tokens(joined), parent_header=parent_header,
                                section_path=section_path, section_depth=section_depth,
                                chunk_type="normal", metadata=metadata.copy()))
        return chunks

    def _chunk_plain_text(self, content, title, metadata):
        path = [title] if title else []
        s = Section(header=title, header_level=1 if title else 0, content=content,
                    start_line=0, end_line=content.count('\n'), full_path=path)
        return self._process_section(s, 0, metadata)

    # ── utilities ────────────────────────────────────────────────────────────

    def _split_paragraphs(self, content: str) -> List[str]:
        return [p.strip() for p in re.split(r'\n{2,}', content) if p.strip()]

    def _get_overlap(self, parts: List[str]) -> str:
        if not parts: return ""
        last = parts[-1]
        if self._estimate_tokens(last) <= self.overlap_tokens:
            return last
        sentences = re.split(r'(?<=[.!?])\s+', last)
        overlap: List[str] = []; count = 0
        for sent in reversed(sentences):
            st = self._estimate_tokens(sent)
            if count + st <= self.overlap_tokens:
                overlap.insert(0, sent); count += st
            else: break
        return ' '.join(overlap)

    def _estimate_tokens(self, text: str) -> int:
        return int(len(text.split()) * 1.3) if text else 0

    def _count_tokens_real(self, text: str) -> Optional[int]:
        if self._tokenizer is None: return None
        try: return len(self._tokenizer.encode(text, add_special_tokens=True))
        except Exception: return None

    def _compute_chunk_quality(self, chunk: Chunk) -> int:
        meta  = chunk.chunk_metadata if chunk.chunk_metadata else chunk.metadata
        score = 50
        if meta.get("ers_markers"):                               score += 15
        if meta.get("entities"):                                  score += min(len(meta["entities"]) * 2, 10)
        if meta.get("api_method") and meta.get("api_endpoint"):   score += 10
        if meta.get("operations"):                                score += min(len(meta["operations"]) * 2, 10)
        if chunk.chunk_type == "atomic_fact":                     score += 5
        if chunk.chunk_type == "document_summary":                score += 5
        if chunk.token_count < 50:                                score -= 10
        return max(0, min(100, score))

    def create_summary_chunk(self, chunks: List[Chunk], document_title: str) -> Optional[Chunk]:
        if len(chunks) < 3: return None
        seen: set = set(); headers: List[str] = []
        for c in chunks:
            if c.parent_header and c.parent_header not in seen:
                headers.append(c.parent_header); seen.add(c.parent_header)
        if not headers: return None
        outline = f"# {document_title}\n\n## Contents\n\n" + "\n".join(f"- {h}" for h in headers[:20])
        return Chunk(content=outline, chunk_index=-1,
                     token_count=int(len(outline.split()) * 1.3),
                     parent_header="Document Outline",
                     section_path=[document_title, "Outline"],
                     section_depth=1, chunk_type="document_summary",
                     metadata={"is_summary": True, "ingestion_version": INGESTION_VERSION,
                                "ers_markers": [], "operations": [], "entities": [],
                                "api_method": None, "api_endpoint": None})

    def get_chunk_distribution(self, chunks: List[Chunk]) -> Dict[str, int]:
        dist = {"tiny": 0, "small": 0, "medium": 0, "large": 0, "oversized": 0}
        for c in chunks:
            t = c.token_count
            if t < 100:   dist["tiny"] += 1
            elif t < 300: dist["small"] += 1
            elif t < 500: dist["medium"] += 1
            elif t <= 800: dist["large"] += 1
            else:          dist["oversized"] += 1
        return dist


# =============================================================================
# DOCUMENT TYPE CLASSIFIER
# =============================================================================

def classify_for_chunking(content: str, source_path: str = "") -> str:
    """Infer the best ChunkingConfig key for a document from path and content."""
    sl = source_path.lower()
    if any(x in sl for x in ("incident", "rca", "postmortem", "outage")):          return "incident"
    if any(x in sl for x in ("runbook", "sop", "procedure", "playbook",
                              "checklist", "howto", "tutorial", "guide")):          return "sop"
    if any(x in sl for x in ("api-reference", "api_reference",
                              "openapi", "endpoint-reference")):                    return "api_reference"
    if any(x in sl for x in ("concept", "overview", "architecture",
                              "theory", "design")):                                 return "concept"

    api_hits  = len(re.findall(r'\b(GET|POST|PUT|DELETE|PATCH)\s+/[A-Za-z0-9_/{}]', content))
    if api_hits >= 4:                                                               return "api_reference"
    step_hits = len(re.findall(r'\b(Step\s+\d+|Phase\s+\d+|\d+\.\s+[A-Z])', content))
    if step_hits >= 3:                                                              return "sop"
    json_hits = len(re.findall(r'"[a-z_]+"\s*:', content))
    if json_hits >= 8:                                                              return "json"
    return "default"


# =============================================================================
# CHUNK ENHANCER
# =============================================================================

class ChunkEnhancer:
    """Adds document-level context metadata to each chunk."""

    def enhance_chunk(self, chunk: Chunk, document_title: str, document_metadata: Dict) -> Chunk:
        parts = []
        if document_title:              parts.append(f"Document: {document_title}")
        if chunk.section_path:          parts.append(f"Section: {' > '.join(chunk.section_path)}")
        chunk.metadata["document_title"]    = document_title
        chunk.metadata["section_path"]      = chunk.section_path
        chunk.metadata["context_prefix"]    = " | ".join(parts)
        chunk.metadata["ingestion_version"] = INGESTION_VERSION
        return chunk
