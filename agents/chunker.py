"""
chunker.py - Structure-aware document chunking for legal texts

This module provides intelligent chunking of legal documents based on their
structure (SCOTUS opinions, Circuit Court decisions, NY Slip Op, BIA decisions).
Chunks are sized to be used in full without truncation.
"""
import re
from typing import List, Dict


def detect_document_type(text: str) -> str:
    """
    Detect the type of legal document based on patterns.

    Returns one of: "scotus", "ny_slip", "circuit", "bia", "generic"
    """
    text_lower = text[:2000].lower()

    if "(slip opinion)" in text_lower or "syllabus" in text_lower:
        return "scotus"
    elif "ny slip op" in text_lower:
        return "ny_slip"
    elif "united states court of appeals" in text_lower:
        return "circuit"
    elif "interim decision" in text_lower or "matter of" in text_lower[:500]:
        return "bia"
    else:
        return "generic"


def chunk_document(text: str, case_id: str, case_name: str,
                   max_chunk_size: int = 2500, overlap: int = 200) -> List[Dict]:
    """
    Chunk a legal document based on its structure.

    Args:
        text: Full document text
        case_id: Unique case identifier
        case_name: Human-readable case name
        max_chunk_size: Target maximum chunk size in characters
        overlap: Number of characters to overlap between chunks

    Returns:
        List of chunk dicts ready for Qdrant insertion, each containing:
        - case_id, case_name, chunk_index, chunk_type, text, char_start, char_end
    """
    if not text or not text.strip():
        return [{
            "case_id": case_id,
            "case_name": case_name,
            "chunk_index": 0,
            "chunk_type": "empty",
            "text": "",
            "char_start": 0,
            "char_end": 0
        }]

    # Small documents: keep as single chunk
    if len(text) < 5000:
        return [{
            "case_id": case_id,
            "case_name": case_name,
            "chunk_index": 0,
            "chunk_type": "full",
            "text": text,
            "char_start": 0,
            "char_end": len(text)
        }]

    doc_type = detect_document_type(text)

    if doc_type == "scotus":
        chunks = chunk_scotus(text, case_id, case_name, max_chunk_size, overlap)
    elif doc_type == "circuit":
        chunks = chunk_circuit(text, case_id, case_name, max_chunk_size, overlap)
    elif doc_type == "ny_slip":
        chunks = chunk_ny_slip(text, case_id, case_name, max_chunk_size, overlap)
    elif doc_type == "bia":
        chunks = chunk_bia(text, case_id, case_name, max_chunk_size, overlap)
    else:
        chunks = chunk_by_paragraphs(text, case_id, case_name, max_chunk_size, overlap)

    # Ensure no chunk exceeds max size
    chunks = ensure_max_chunk_size(chunks, case_id, case_name, max_chunk_size, overlap)

    return chunks if chunks else [{
        "case_id": case_id,
        "case_name": case_name,
        "chunk_index": 0,
        "chunk_type": "full",
        "text": text[:max_chunk_size],
        "char_start": 0,
        "char_end": min(len(text), max_chunk_size)
    }]


def ensure_max_chunk_size(chunks: List[Dict], case_id: str, case_name: str,
                          max_size: int = 2500, overlap: int = 200) -> List[Dict]:
    """
    Recursively split any chunks that exceed max_size.
    Ensures no chunk is too large to use in full.
    """
    result = []
    for chunk in chunks:
        if len(chunk["text"]) <= max_size:
            chunk["chunk_index"] = len(result)
            result.append(chunk)
        else:
            # Split large chunk into smaller pieces using character-based splitting
            text = chunk["text"]
            chunk_type = chunk.get("chunk_type", "paragraph")
            char_offset = chunk.get("char_start", 0)

            # Split by finding good break points (sentences, then words)
            sub_texts = []
            current_pos = 0

            while current_pos < len(text):
                # Find the end position for this sub-chunk
                end_pos = min(current_pos + max_size, len(text))

                if end_pos < len(text):
                    # Try to find a sentence boundary
                    search_start = max(current_pos + max_size - 500, current_pos)
                    last_sentence = max(
                        text.rfind('. ', search_start, end_pos),
                        text.rfind('? ', search_start, end_pos),
                        text.rfind('! ', search_start, end_pos)
                    )
                    if last_sentence > current_pos:
                        end_pos = last_sentence + 2
                    else:
                        # Fall back to word boundary
                        last_space = text.rfind(' ', search_start, end_pos)
                        if last_space > current_pos:
                            end_pos = last_space + 1

                sub_text = text[current_pos:end_pos].strip()
                if sub_text:
                    sub_texts.append((current_pos, end_pos, sub_text))

                # Move to next chunk with overlap
                current_pos = max(end_pos - overlap, current_pos + 1)
                if current_pos >= len(text):
                    break

            # Create sub-chunks
            for i, (start, end, sub_text) in enumerate(sub_texts):
                result.append({
                    "case_id": case_id,
                    "case_name": case_name,
                    "chunk_index": len(result),
                    "chunk_type": f"{chunk_type}_part{i}",
                    "text": sub_text,
                    "char_start": char_offset + start,
                    "char_end": char_offset + end
                })

    return result


def chunk_scotus(text: str, case_id: str, case_name: str,
                 max_chunk_size: int, overlap: int) -> List[Dict]:
    """
    Chunk Supreme Court opinions by syllabus and sections.

    SCOTUS opinions have clear structure:
    - Syllabus (summary)
    - "Held:" section (key holding)
    - Opinion of the Court (analysis)
    """
    chunks = []

    # Find syllabus section (high-value summary)
    syllabus_match = re.search(
        r'Syllabus(.*?)(?=Opinion of the Court|OPINION OF THE COURT|Cite as:|$)',
        text, re.DOTALL | re.IGNORECASE
    )
    if syllabus_match:
        syllabus_text = syllabus_match.group(0).strip()
        chunks.append({
            "case_id": case_id,
            "case_name": case_name,
            "chunk_index": 0,
            "chunk_type": "syllabus",
            "text": syllabus_text,
            "char_start": syllabus_match.start(),
            "char_end": syllabus_match.end()
        })

    # Find "Held:" section (key holding - very high value)
    held_match = re.search(r'Held:(.*?)(?=Pp\.\s*\d|Opinion of|OPINION OF|$)', text, re.DOTALL)
    if held_match:
        held_text = "Held:" + held_match.group(1).strip()
        chunks.append({
            "case_id": case_id,
            "case_name": case_name,
            "chunk_index": len(chunks),
            "chunk_type": "holding",
            "text": held_text,
            "char_start": held_match.start(),
            "char_end": held_match.end()
        })

    # Find and chunk the Opinion of the Court
    opinion_match = re.search(r'(Opinion of the Court|OPINION OF THE COURT)', text, re.IGNORECASE)
    if opinion_match:
        opinion_start = opinion_match.start()
        opinion_text = text[opinion_start:]

        # Try to split by section markers (I, II, III or A, B, C)
        section_pattern = r'\n\s*([IVX]+\.?\s|[A-C]\.\s)'
        sections = re.split(section_pattern, opinion_text)

        if len(sections) > 2:
            current_pos = opinion_start
            for i in range(0, len(sections), 2):
                section_text = sections[i]
                if i + 1 < len(sections):
                    section_text = sections[i + 1] + section_text if i > 0 else section_text

                if len(section_text.strip()) > 100:
                    chunks.append({
                        "case_id": case_id,
                        "case_name": case_name,
                        "chunk_index": len(chunks),
                        "chunk_type": "analysis",
                        "text": section_text.strip(),
                        "char_start": current_pos,
                        "char_end": current_pos + len(section_text)
                    })
                current_pos += len(section_text)
        else:
            # No clear sections, chunk by paragraphs
            para_chunks = chunk_by_paragraphs(opinion_text, case_id, case_name,
                                              max_chunk_size, overlap)
            for chunk in para_chunks:
                chunk["chunk_type"] = "analysis"
                chunk["char_start"] += opinion_start
                chunk["char_end"] += opinion_start
                chunks.append(chunk)

    return chunks if chunks else chunk_by_paragraphs(text, case_id, case_name,
                                                      max_chunk_size, overlap)


def chunk_circuit(text: str, case_id: str, case_name: str,
                  max_chunk_size: int, overlap: int) -> List[Dict]:
    """
    Chunk Circuit Court opinions by section headers.

    Common sections: BACKGROUND, DISCUSSION, ANALYSIS, STANDARD OF REVIEW, CONCLUSION
    """
    chunks = []

    # Extract header info first
    header_match = re.search(
        r'^(.*?)(BACKGROUND|DISCUSSION|ANALYSIS|I\.\s)',
        text, re.DOTALL | re.IGNORECASE
    )
    if header_match:
        header_text = header_match.group(1).strip()
        if len(header_text) > 100:
            chunks.append({
                "case_id": case_id,
                "case_name": case_name,
                "chunk_index": 0,
                "chunk_type": "header",
                "text": header_text,
                "char_start": 0,
                "char_end": len(header_text)
            })

    # Look for common section markers
    section_pattern = r'\n\s*(BACKGROUND|DISCUSSION|ANALYSIS|STANDARD OF REVIEW|CONCLUSION|FACTUAL AND PROCEDURAL HISTORY|I\.|II\.|III\.|IV\.)'

    # Split and keep delimiters
    parts = re.split(f'({section_pattern})', text, flags=re.IGNORECASE)

    current_pos = 0
    i = 0
    while i < len(parts):
        part = parts[i]

        # Check if this is a section header
        if re.match(section_pattern, '\n' + part.strip(), re.IGNORECASE):
            # Combine header with following content
            section_header = part.strip()
            section_content = parts[i + 1] if i + 1 < len(parts) else ""
            section_text = section_header + section_content

            if len(section_text.strip()) > 100:
                # Determine chunk type from header
                header_lower = section_header.lower()
                if 'background' in header_lower or 'factual' in header_lower:
                    chunk_type = "background"
                elif 'discussion' in header_lower or 'analysis' in header_lower:
                    chunk_type = "analysis"
                elif 'conclusion' in header_lower:
                    chunk_type = "conclusion"
                else:
                    chunk_type = "section"

                chunks.append({
                    "case_id": case_id,
                    "case_name": case_name,
                    "chunk_index": len(chunks),
                    "chunk_type": chunk_type,
                    "text": section_text.strip(),
                    "char_start": current_pos,
                    "char_end": current_pos + len(section_text)
                })
            current_pos += len(section_text)
            i += 2
        else:
            current_pos += len(part)
            i += 1

    # If no sections found, use paragraph chunking
    if len(chunks) <= 1:
        return chunk_by_paragraphs(text, case_id, case_name, max_chunk_size, overlap)

    return chunks


def chunk_ny_slip(text: str, case_id: str, case_name: str,
                  max_chunk_size: int, overlap: int) -> List[Dict]:
    """
    Chunk NY Slip Op decisions.

    These often have a header section followed by the opinion body.
    """
    chunks = []

    # Find the boilerplate end marker
    boilerplate_end = re.search(
        r'Published by New York State Law Reporting Bureau.*?(?=\n\n|\. [A-Z])',
        text, re.DOTALL
    )

    if boilerplate_end:
        # Header chunk
        header_end = boilerplate_end.end()
        header_text = text[:header_end].strip()

        if len(header_text) > 100:
            chunks.append({
                "case_id": case_id,
                "case_name": case_name,
                "chunk_index": 0,
                "chunk_type": "header",
                "text": header_text,
                "char_start": 0,
                "char_end": header_end
            })

        # Body text
        body_text = text[header_end:].strip()
        body_chunks = chunk_by_paragraphs(body_text, case_id, case_name,
                                          max_chunk_size, overlap)
        for chunk in body_chunks:
            chunk["chunk_type"] = "opinion"
            chunk["char_start"] += header_end
            chunk["char_end"] += header_end
            chunks.append(chunk)
    else:
        # No clear structure, use paragraph chunking
        return chunk_by_paragraphs(text, case_id, case_name, max_chunk_size, overlap)

    return chunks


def chunk_bia(text: str, case_id: str, case_name: str,
              max_chunk_size: int, overlap: int) -> List[Dict]:
    """
    Chunk BIA (Board of Immigration Appeals) decisions.

    These often have numbered holdings and structured analysis.
    """
    chunks = []

    # Find the header/citation section
    header_match = re.search(
        r'^(.*?Decided by Board.*?\d{4})',
        text, re.DOTALL
    )

    if header_match:
        header_text = header_match.group(1).strip()
        if len(header_text) > 100:
            chunks.append({
                "case_id": case_id,
                "case_name": case_name,
                "chunk_index": 0,
                "chunk_type": "header",
                "text": header_text,
                "char_start": 0,
                "char_end": header_match.end()
            })

    # Look for numbered holdings (1., 2., 3.)
    holding_pattern = r'\n\s*(\d+)\.\s+'
    parts = re.split(holding_pattern, text)

    if len(parts) > 2:
        current_pos = len(parts[0])
        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                number = parts[i]
                content = parts[i + 1]
                holding_text = f"{number}. {content}".strip()

                if len(holding_text) > 100:
                    chunks.append({
                        "case_id": case_id,
                        "case_name": case_name,
                        "chunk_index": len(chunks),
                        "chunk_type": f"holding_{number}",
                        "text": holding_text,
                        "char_start": current_pos,
                        "char_end": current_pos + len(holding_text)
                    })
                current_pos += len(number) + len(content) + 3
    else:
        # No numbered sections, use paragraph chunking
        return chunk_by_paragraphs(text, case_id, case_name, max_chunk_size, overlap)

    return chunks if chunks else chunk_by_paragraphs(text, case_id, case_name,
                                                      max_chunk_size, overlap)


def chunk_by_paragraphs(text: str, case_id: str, case_name: str,
                        max_chunk_size: int, overlap: int) -> List[Dict]:
    """
    Generic paragraph-based chunking with overlap.

    This is the fallback for documents without clear structure.
    Splits at paragraph boundaries and maintains overlap for context.
    """
    # Split by double newlines (paragraphs) OR single newlines if no double newlines
    paragraphs = re.split(r'\n\s*\n', text)

    # If we didn't get meaningful paragraphs, try splitting by single newlines
    if len(paragraphs) <= 1 or (len(paragraphs) < 5 and len(text) > 5000):
        paragraphs = text.split('\n')

    # If still too few paragraphs, split by sentences
    if len(paragraphs) <= 1 or (len(paragraphs) < 5 and len(text) > 5000):
        paragraphs = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = ""
    current_start = 0
    char_pos = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            char_pos += 2  # Account for newlines
            continue

        # Check if adding this paragraph would exceed max size
        if len(current_chunk) + len(para) + 2 > max_chunk_size:
            if current_chunk:
                chunks.append({
                    "case_id": case_id,
                    "case_name": case_name,
                    "chunk_index": len(chunks),
                    "chunk_type": "paragraph",
                    "text": current_chunk.strip(),
                    "char_start": current_start,
                    "char_end": char_pos
                })

            # Start new chunk with overlap from previous
            if len(current_chunk) > overlap:
                # Find a good break point in the overlap region
                overlap_text = current_chunk[-overlap:]
                # Try to start at a sentence boundary
                sentence_break = overlap_text.rfind('. ')
                if sentence_break > 0:
                    overlap_text = overlap_text[sentence_break + 2:]
            else:
                overlap_text = ""

            current_chunk = overlap_text + ("\n\n" if overlap_text else "") + para
            current_start = char_pos - len(overlap_text)
        else:
            current_chunk += ("\n\n" if current_chunk else "") + para

        char_pos += len(para) + 2

    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append({
            "case_id": case_id,
            "case_name": case_name,
            "chunk_index": len(chunks),
            "chunk_type": "paragraph",
            "text": current_chunk.strip(),
            "char_start": current_start,
            "char_end": char_pos
        })

    return chunks


# Convenience function for testing
def analyze_document(text: str) -> Dict:
    """
    Analyze a document and return info about how it would be chunked.
    Useful for testing and debugging.
    """
    doc_type = detect_document_type(text)
    chunks = chunk_document(text, "test_id", "Test Case")

    return {
        "doc_type": doc_type,
        "total_chars": len(text),
        "num_chunks": len(chunks),
        "chunk_types": [c["chunk_type"] for c in chunks],
        "chunk_sizes": [len(c["text"]) for c in chunks],
        "avg_chunk_size": sum(len(c["text"]) for c in chunks) / len(chunks) if chunks else 0
    }
