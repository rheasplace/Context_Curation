"""Offline contract checks; run with python -m unittest -v."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from universal_context import load_documents, extract_universal_context, split_text
from level2_vector_rag import index_folder, retrieve_context


class PipelineTests(unittest.TestCase):
    def test_recursive_mixed_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "nested").mkdir()
            (root / "notes.md").write_text("Root notes")
            (root / "nested/notes.md").write_text("Nested notes")
            (root / "nested/picture.PNG").write_bytes(b"fake image")
            (root / "nested/pages.pdf").write_bytes(b"fake PDF")
            (root / ".hidden.md").write_text("Excluded")
            (root / "link.md").symlink_to(root / "notes.md")
            vision = Mock(return_value={"message": {"content": "A melting clock"}})
            reader = Mock(return_value=SimpleNamespace(pages=[
                SimpleNamespace(extract_text=lambda: "First page"),
                SimpleNamespace(extract_text=lambda: "Second page"),
            ]))
            with patch.dict(sys.modules, {"ollama": SimpleNamespace(chat=vision),
                                          "pypdf": SimpleNamespace(PdfReader=reader)}):
                docs = load_documents(folder)
                context = extract_universal_context(folder)
            self.assertEqual(len(docs), 5)
            self.assertEqual([d.page for d in docs if d.kind == "pdf"], [1, 2])
            self.assertIn("nested/notes.md", context)
            self.assertIn("A melting clock", context)
            self.assertNotIn("Excluded", context)

    def test_chunk_coverage_and_bounds(self):
        text = "".join(chr(0x400 + i) for i in range(1400))
        chunks = list(split_text(text, 600, 80))
        self.assertTrue(all(len(c) <= 600 for c in chunks))
        self.assertEqual(chunks[0] + "".join(c[80:] for c in chunks[1:]), text)
        paragraphs = list(split_text("a" * 400 + "\n\n" + "b" * 400))
        self.assertEqual(paragraphs[0], "a" * 400)
        with self.assertRaises(ValueError):
            list(split_text("text", 80, 80))

    def test_index_and_query_embedding_contract(self):
        collection = Mock()
        collection.get.return_value = {"ids": ["obsolete"]}
        embed = Mock(return_value={"embeddings": [[0.1, 0.2, 0.3]]})
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "a").mkdir()
            (root / "notes.md").write_text("Top level")
            (root / "a/notes.md").write_text("Nested")
            with patch.dict(sys.modules, {"ollama": SimpleNamespace(embed=embed)}):
                self.assertEqual(index_folder(collection, folder), 2)
                calls = collection.upsert.call_args_list
                self.assertNotEqual(calls[0].kwargs["ids"], calls[1].kwargs["ids"])
                self.assertEqual(calls[0].kwargs["embeddings"], [[0.1, 0.2, 0.3]])
                meta = calls[0].kwargs["metadatas"][0]
                collection.query.return_value = {"documents": [["Evidence"]], "metadatas": [[meta]]}
                self.assertIn("Evidence", retrieve_context(collection, "Question", folder, 3))
                self.assertEqual(collection.query.call_args.kwargs["n_results"], 1)
                self.assertEqual(collection.query.call_args.kwargs["query_embeddings"], [[0.1, 0.2, 0.3]])
                collection.delete.assert_called_once_with(ids=["obsolete"])
                collection.reset_mock()
                embed.side_effect = RuntimeError("Model unavailable")
                with self.assertRaises(RuntimeError):
                    index_folder(collection, folder)
                collection.delete.assert_not_called()


if __name__ == "__main__":
    unittest.main()
