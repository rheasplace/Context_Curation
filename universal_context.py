"""Shared recursive extraction and bounded text chunking for the local demos."""

from dataclasses import dataclass
from pathlib import Path
import re
import warnings


@dataclass(frozen=True)
class Document:
    source: str
    kind: str
    text: str
    page: int = 0


def load_documents(folder_path, vision_model="llava"):
    """Read UTF-8 text, PDF pages, and image descriptions; fail on read errors.

    Hidden paths and symlinks are excluded. Scanned PDFs require separate OCR.
    """
    root = Path(folder_path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Create a context directory first: {root}")
    documents = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part.startswith(".") for part in relative.parts):
            continue
        if any(p.is_symlink() for p in [path, *path.parents] if p != root):
            continue
        if not path.is_file():
            continue
        source = relative.as_posix()
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md"}:
            documents.append(Document(source, "text", path.read_text(encoding="utf-8")))
        elif suffix == ".pdf":
            from pypdf import PdfReader
            for page_number, page in enumerate(PdfReader(path).pages, 1):
                text = page.extract_text() or ""
                if not text.strip():
                    warnings.warn(f"No text in {source}, page {page_number}; OCR may be needed.")
                documents.append(Document(source, "pdf", text, page_number))
        elif suffix in {".jpg", ".jpeg", ".png"}:
            import ollama
            response = ollama.chat(model=vision_model, messages=[{
                "role": "user",
                "content": "Describe this image in detail and extract visible text. State uncertainty.",
                "images": [str(path)],
            }])
            documents.append(Document(source, "image", response["message"]["content"]))
    return [doc for doc in documents if doc.text.strip()]


def source_label(document):
    page = f", page {document.page}" if document.page else ""
    return f"{document.source}{page} ({document.kind})"


def extract_universal_context(folder_path, vision_model="llava"):
    return "\n\n".join(
        f"--- Source: {source_label(doc)} ---\n{doc.text}"
        for doc in load_documents(folder_path, vision_model)
    )


def load_corpus_documents(folder_path):
    """Read every supported file, preserving source boundaries in Level 1 snapshots.

    Snapshot sources use prepared-file::original-source labels. Plain files retain
    their relative paths. Do not reconstruct or reopen original source paths.
    """
    header = re.compile(r"^--- Source: (.+?)(?:, page (\d+))? \((text|pdf|image)\) ---\s*$", re.MULTILINE)
    documents = []
    for doc in load_documents(folder_path):
        matches = list(header.finditer(doc.text)) if doc.kind == "text" else []
        if not matches or doc.text[:matches[0].start()].strip():
            documents.append(doc)
            continue
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(doc.text)
            text = doc.text[match.end():end].strip()
            if text:
                documents.append(Document(f"{doc.source}::{match[1]}", match[3], text, int(match[2] or 0)))
    return documents


def split_text(text, chunk_size=600, overlap=80):
    """Prefer paragraph/word boundaries, with overlap within each source unit.

    Sizes are characters, not tokens. Embedding uses truncate=False so oversized
    inputs fail explicitly; reduce chunk_size if the embedding model rejects one.
    """
    if not 0 <= overlap < chunk_size:
        raise ValueError("Require 0 <= overlap < chunk_size")
    text = text.strip()
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            lower = start + max(chunk_size // 2, overlap + 1)
            paragraph = text.rfind("\n\n", lower, end)
            if paragraph != -1:
                end = paragraph + 2
            else:
                boundaries = list(re.finditer(r"\s+", text[lower:end]))
                if boundaries:
                    end = lower + boundaries[-1].end()
        chunk = text[start:end].strip()
        if chunk:
            yield chunk
        if end == len(text):
            break
        start = end - overlap
