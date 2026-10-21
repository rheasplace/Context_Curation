"""Offline tests of extraction, provenance, atomic publication, and graph retrieval."""
import json
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from build_graph import build, prepare_batches, validate_edges
from graph_context import load_graph, retrieve_relationships
from model_providers import Answer
from universal_context import load_corpus_documents


def extraction(*args, **kwargs):
    items = json.loads(args[2].split("EXCERPTS:\n", 1)[1])
    return Answer(json.dumps({"relationships": [
        {"source_id": item["id"], "subject": "Ada", "relation": "developed", "object": "Algorithm",
         "quote": item["text"]} for item in items]}), "STOP", {"candidates_token_count": 20}, "test-model")


class GraphTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.folder = self.root / "prepared_context"
        self.folder.mkdir()
        self.graph = self.root / "graph.json"
        (self.folder / "course.txt").write_text("--- Source: original.pdf, page 2 (pdf) ---\nAda developed Algorithm.\n\n--- Source: figure.png (image) ---\nAda developed Algorithm.")
        (self.folder / "nested").mkdir()
        (self.folder / "nested/note.md").write_text("Ada developed Algorithm.")

    def test_snapshot_sources_and_all_nested_content(self):
        docs = load_corpus_documents(self.folder)
        self.assertEqual(len(docs), 3)
        self.assertEqual((docs[0].source, docs[0].page), ("course.txt::original.pdf", 2))
        self.assertEqual(docs[1].kind, "image")
        self.assertEqual(docs[2].source, "nested/note.md")
        self.assertEqual(sum(len(batch) for batch in prepare_batches(docs, 6000, 300)), 3)

    def test_extraction_cache_dedup_and_retrieval(self):
        generate = Mock(side_effect=extraction)
        artifact = build(self.folder, self.graph, generate=generate)
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(len(artifact["relationships"]), 1)
        self.assertEqual(len(artifact["relationships"][0]["evidence"]), 3)
        build(self.folder, self.graph, generate=generate)
        self.assertEqual(generate.call_count, 1)
        result = retrieve_relationships(self.graph, self.folder, entity="ada")
        self.assertFalse(result["illustrative"])
        self.assertEqual(len(result["relationships"]), 1)
        self.assertEqual(retrieve_relationships(self.graph, self.folder, query="unrelated topic")["relationships"], [])
        self.assertEqual(retrieve_relationships(self.graph, self.folder, query="Who developed Algorithm?")["total_matches"], 1)
        (self.folder / "nested/note.md").write_text("Changed")
        with self.assertRaisesRegex(ValueError, "stale"):
            load_graph(self.graph, self.folder)

    def test_failed_build_preserves_graph_and_resumes_cached_batches(self):
        self.graph.write_text("previous graph")
        count = 0
        def fail_second(*args, **kwargs):
            nonlocal count
            count += 1
            if count == 2:
                return Answer("partial", "MAX_TOKENS")
            return extraction(*args, **kwargs)
        with self.assertRaisesRegex(ValueError, "MAX_TOKENS"):
            build(self.folder, self.graph, chunk_size=100, overlap=10, generate=fail_second)
        self.assertEqual(self.graph.read_text(), "previous graph")
        generate = Mock(side_effect=extraction)
        result = build(self.folder, self.graph, chunk_size=100, overlap=10, generate=generate)
        self.assertEqual(generate.call_count, 2)
        self.assertTrue(result["complete"])
        other = self.root / "other"
        other.mkdir()
        with self.assertRaisesRegex(ValueError, "different corpus"):
            load_graph(self.graph, other)

    def test_graph_and_hybrid_cli_use_persisted_evidence(self):
        import level3_graph_rag
        import level4_hybrid_rag
        build(self.folder, self.graph, generate=extraction)
        argv = ["demo", "--folder", str(self.folder), "--graph", str(self.graph),
                "--query", "Who developed Algorithm?"]
        with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()), \
                patch("level3_graph_rag.run_answers", return_value=0) as answer:
            self.assertEqual(level3_graph_rag.main(), 0)
            self.assertIn("course.txt::original.pdf", answer.call_args.args[0])
            self.assertIn("Ada developed Algorithm.", answer.call_args.args[0])
        with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()), \
                patch("level2_vector_rag.get_collection"), \
                patch("level2_vector_rag.retrieve_context", return_value="Vector evidence"), \
                patch("level4_hybrid_rag.run_answers", return_value=0) as answer:
            self.assertEqual(level4_hybrid_rag.main(), 0)
            self.assertIn("Vector evidence", answer.call_args.args[0])
            self.assertIn("Ada developed Algorithm.", answer.call_args.args[0])

    def test_unverifiable_quote_or_unknown_source_rejected(self):
        edge = {"source_id": "s0", "subject": "Ada", "relation": "developed", "object": "Algorithm", "quote": "Invented quote"}
        with self.assertRaisesRegex(ValueError, "exact supporting"):
            validate_edges(json.dumps({"relationships": [edge]}), {"s0": "Ada developed Algorithm."})
        edge["quote"] = "Ada developed Algorithm."
        with self.assertRaises(ValueError):
            validate_edges(json.dumps({"relationships": [edge]}), {"other": edge["quote"]})
        with self.assertRaises(ValueError):
            build(self.folder, self.folder / "graph.json", generate=extraction)


if __name__ == "__main__":
    unittest.main()
