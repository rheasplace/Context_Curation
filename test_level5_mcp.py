"""Real Chroma and MCP transport checks using synthetic documents; no model calls."""
import asyncio
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

import chromadb
from chromadb.config import Settings
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from corpus_tools import CorpusTools
from graph_context import atomic_json, corpus_fingerprint
from level2_vector_rag import COLLECTION, EMBED_MODEL


class CorpusTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.folder = self.root / "sources"
        self.folder.mkdir()
        self.db = self.root / "db"
        self.graph = self.root / "graph.json"
        (self.folder / "note.txt").write_text("Freud influenced Dalí.")
        atomic_json(self.graph, {"version": 1, "complete": True, "corpus": str(self.folder),
            "corpus_sha256": corpus_fingerprint(self.folder), "model": "mock-gemini",
            "relationships": [{"subject": "Freud", "relation": "influenced", "object": "Dalí",
                "evidence": [{"source": "note.txt", "page": 0, "kind": "text", "chunk": 0,
                              "quote": "Freud influenced Dalí."}]}]})
        client = chromadb.PersistentClient(path=str(self.db), settings=Settings(anonymized_telemetry=False))
        self.collection = client.create_collection(COLLECTION, embedding_function=None)
        metas = [{"corpus": str(self.folder), "source": "sub/note.pdf", "kind": "pdf",
                  "page": 1, "chunk": i, "embedding_model": EMBED_MODEL} for i in range(3)]
        metas += [dict(metas[0], page=2), dict(metas[0], corpus=str(self.root / "other"))]
        self.collection.add(ids=["a", "b", "c", "page2", "foreign"],
                            documents=["First", "Middle", "Last", "Other page", "Private"],
                            metadatas=metas, embeddings=[[1., 0., 0.]] * 5)
        self.embed = Mock(return_value={"embeddings": [[1., 0., 0.]]})
        self.service = CorpusTools(self.folder, self.db, self.collection, self.embed, self.graph)

    def test_search_is_scoped_and_bounded(self):
        matches = self.service.search_context("evidence", 10)["matches"]
        self.assertEqual(len(matches), 4)
        self.assertNotIn("foreign", [m["chunk_id"] for m in matches])
        self.embed.assert_called_once_with(model=EMBED_MODEL, input="evidence", truncate=False)
        for query, limit in [("", 3), ("x", 11), ("x" * 4001, 3)]:
            with self.assertRaises(ValueError):
                self.service.search_context(query, limit)

    def test_excerpts_do_not_cross_corpus_or_page(self):
        self.assertEqual([r["chunk_id"] for r in self.service.fetch_source_excerpt("b")["chunks"]], ["a", "b", "c"])
        with self.assertRaises(ValueError):
            self.service.fetch_source_excerpt("foreign")
        with self.assertRaises(ValueError):
            self.service.fetch_source_excerpt("b", 3)
        record = self.service.record("x", "x" * 5000, {"source": "a", "page": 1, "kind": "txt", "chunk": 0})
        self.assertTrue(record["truncated"])
        self.assertEqual(len(record["text"]), 4000)

    def test_status_sources_graph_and_missing_index(self):
        self.assertEqual(self.service.get_corpus_status()["indexed_chunks"], 4)
        self.assertEqual(self.service.list_sources()["sources"][0]["chunks"], 4)
        self.assertEqual(self.service.list_sources(offset=1)["sources"], [])
        self.assertFalse(self.service.get_relationships("Dalí")["illustrative"])
        missing = self.root / "missing"
        self.assertFalse(CorpusTools(self.folder, missing).get_corpus_status()["ready"])
        self.assertFalse(missing.exists())

    def test_real_stdio_handshake_and_tools(self):
        async def check():
            params = StdioServerParameters(command=sys.executable, args=[
                str(Path(__file__).parent / "level5_mcp_server.py"),
                "--folder", str(self.folder), "--db", str(self.db), "--graph", str(self.graph)])
            async with stdio_client(params) as (reader, writer):
                async with ClientSession(reader, writer) as session:
                    await session.initialize()
                    tools = (await session.list_tools()).tools
                    self.assertEqual({t.name for t in tools}, {"get_corpus_status", "list_sources", "search_context", "fetch_source_excerpt", "get_relationships"})
                    self.assertTrue(all(t.annotations.readOnlyHint for t in tools))
                    status = await session.call_tool("get_corpus_status", {})
                    self.assertFalse(status.isError)
                    self.assertEqual(status.structuredContent["indexed_chunks"], 4)
                    excerpt = await session.call_tool("fetch_source_excerpt", {"chunk_id": "b"})
                    self.assertEqual(len(excerpt.structuredContent["chunks"]), 3)
                    foreign = await session.call_tool("fetch_source_excerpt", {"chunk_id": "foreign"})
                    self.assertTrue(foreign.isError)
                    graph = await session.call_tool("get_relationships", {"entity": "Dalí"})
                    self.assertFalse(graph.structuredContent["illustrative"])
                    self.assertEqual(graph.structuredContent["relationships"][0]["evidence"][0]["source"], "note.txt")
        asyncio.run(asyncio.wait_for(check(), timeout=30))


if __name__ == "__main__":
    unittest.main()
