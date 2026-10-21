"""Persist and retrieve source-backed relationships extracted from the prepared corpus."""
import hashlib
import json
from pathlib import Path
import re
import tempfile

import networkx as nx

SUPPORTED = {".txt", ".md", ".pdf", ".png", ".jpg", ".jpeg"}
STOPWORDS = set("a an the of in on at to for with and or is are was were be what who how why which me tell about their they it from does do key themes related cite sources".split())


def corpus_fingerprint(folder):
    root = Path(folder).resolve()
    if not root.is_dir():
        raise ValueError(f"Corpus folder does not exist: {root}")
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(p.startswith(".") for p in relative.parts) or path.suffix.lower() not in SUPPORTED:
            continue
        if any(p.is_symlink() for p in [path, *path.parents] if p != root) or not path.is_file():
            continue
        digest.update(relative.as_posix().encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2)
        temporary.replace(path)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()


def load_graph(path, folder):
    path = Path(path)
    if not path.is_file():
        raise ValueError("Graph not found. Run build_graph.py with the same --folder and --graph first.")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != 1 or not data.get("complete"):
        raise ValueError("Graph is incomplete or unsupported; rebuild it with build_graph.py.")
    if data.get("corpus") != str(Path(folder).resolve()):
        raise ValueError("Graph belongs to a different corpus; select the matching --folder/--graph.")
    if data.get("corpus_sha256") != corpus_fingerprint(folder):
        raise ValueError("Graph is stale. Rebuild graph and vector index after prepared context changes.")
    graph = nx.MultiDiGraph()
    for edge in data["relationships"]:
        graph.add_edge(edge["subject"], edge["object"], **edge)
    return data, graph


def terms(text):
    return {word for word in re.findall(r"[^\W_]+", text.casefold()) if word not in STOPWORDS and len(word) > 1}


def retrieve_relationships(path, folder, entity="", query="", limit=12):
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 20:
        raise ValueError("limit must be an integer from 1 to 20")
    if len(entity) > 200 or len(query) > 4000:
        raise ValueError("entity/query exceed their 200/4000 character limits")
    data, graph = load_graph(path, folder)
    selector = entity.strip().casefold()
    nodes = {n for n in graph if selector and (selector == n.casefold() or selector in n.casefold())}
    words = terms(query)
    ranked = []
    for source, target, attrs in graph.edges(data=True):
        if selector and source not in nodes and target not in nodes:
            continue
        score = len(words & terms(f"{source} {attrs['relation']} {target} " +
                                 " ".join(e['quote'] for e in attrs['evidence'])))
        if not selector and words and not score:
            continue
        ranked.append((score, attrs))
    ranked.sort(key=lambda item: (-item[0], item[1]["subject"], item[1]["relation"], item[1]["object"]))
    edges = [dict(edge, evidence=edge["evidence"][:3]) for _, edge in ranked[:limit]]
    return {"relationships": edges, "total_matches": len(ranked), "truncated": len(ranked) > limit,
            "entities": sorted(graph.nodes)[:50], "total_entities": graph.number_of_nodes(),
            "provenance": "Gemini-extracted claims from prepared sources; quotes checked against input, semantics require review",
            "model": data["model"], "corpus_sha256": data["corpus_sha256"], "illustrative": False}


def get_graph_context(path, folder, entity="", query="", limit=12):
    result = retrieve_relationships(path, folder, entity, query, limit)
    return json.dumps(result, ensure_ascii=False, indent=2)
