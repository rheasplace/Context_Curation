"""Offline adapter and CLI integration checks; no API requests."""
import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import MagicMock, Mock, patch

import level1_raw_context
import level2_vector_rag
from model_providers import (Answer, add_model_arguments, generate_answer,
                             run_answers, selected_models, output_limit, gemini_thinking_config)


def arguments(*argv):
    parser = argparse.ArgumentParser()
    add_model_arguments(parser)
    return parser.parse_args(argv)


class ProviderTests(unittest.TestCase):
    def test_output_and_thinking_defaults(self):
        self.assertEqual(output_limit("gemini"), 8192)
        self.assertEqual(output_limit("ollama"), 1024)
        self.assertEqual(output_limit("gemini", 16000), 16000)
        self.assertEqual(gemini_thinking_config("gemini-3.8-flash"), {"thinking_level": "low"})
        self.assertEqual(gemini_thinking_config("gemini-2.5-flash"), {"thinking_budget": 1024})
        self.assertEqual(gemini_thinking_config("gemini-3.8-flash", "provider-default"), {})
        with self.assertRaises(ValueError):
            gemini_thinking_config("gemini-3.8-flash", budget=100)
        from google.genai import types
        config = types.GenerateContentConfig(max_output_tokens=8192,
                    thinking_config=gemini_thinking_config("gemini-3.8-flash"))
        self.assertEqual(config.thinking_config.thinking_level.value, "LOW")

    def test_empty_max_tokens_saved_without_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = Path(tmp) / "result.json"
            args = arguments("--provider", "gemini", "--results", str(result))
            answer = Answer("", "MAX_TOKENS", {"thoughts_token_count": 8192},
                            request_settings={"max_output_tokens": 8192})
            output = io.StringIO()
            with redirect_stdout(output), patch("model_providers.generate_answer", return_value=answer) as call:
                self.assertEqual(run_answers("prompt", args), 1)
            call.assert_called_once()
            row = json.loads(result.read_text())["runs"][0]
            self.assertEqual(row["status"], "incomplete")
            self.assertEqual(row["effective_max_output_tokens"], 8192)
            self.assertEqual(row["request_settings"]["max_output_tokens"], 8192)
            self.assertIn("Thinking: 8192", output.getvalue())

    def test_index_only_does_not_generate_or_retrieve(self):
        with patch.object(sys, "argv", ["level2", "--index-only"]), redirect_stdout(io.StringIO()), \
                patch("level2_vector_rag.get_collection"), patch("level2_vector_rag.index_folder", return_value=2), \
                patch("level2_vector_rag.retrieve_context") as retrieve, patch("level2_vector_rag.run_answers") as run:
            self.assertEqual(level2_vector_rag.main(), 0)
            retrieve.assert_not_called()
            run.assert_not_called()

    def test_defaults_and_selection(self):
        self.assertEqual(selected_models(arguments()), [("ollama", "phi4-mini:3.8b-q4_K_M")])
        self.assertEqual(selected_models(arguments("--provider", "gemini")), [("gemini", "gemini-3.8-flash")])
        for argv in [("--provider", "openai"), ("--compare", "unknown=x"),
                     ("--compare", "gemini="), ("--compare", "gemini=x", "--model", "y")]:
            with self.assertRaises(ValueError):
                selected_models(arguments(*argv))

    def test_missing_key_fails_before_loading_sdk(self):
        with patch.dict(os.environ, {}, clear=True):
            for provider, key in [("gemini", "GEMINI_API_KEY"), ("openai", "OPENAI_API_KEY"),
                                  ("anthropic", "ANTHROPIC_API_KEY")]:
                with self.assertRaisesRegex(ValueError, key):
                    generate_answer(provider, "model", "prompt")

    def test_gemini_stage_default_preserves_explicit_overrides(self):
        parser = argparse.ArgumentParser()
        add_model_arguments(parser, default_provider="gemini")
        self.assertEqual(selected_models(parser.parse_args([])), [("gemini", "gemini-3.8-flash")])
        self.assertEqual(selected_models(parser.parse_args(["--provider", "ollama"])),
                         [("ollama", "phi4-mini:3.8b-q4_K_M")])
        self.assertEqual(selected_models(parser.parse_args(["--compare", "ollama=mini"])),
                         [("ollama", "mini")])

    def test_ollama_contract(self):
        client = Mock()
        client.generate.return_value = {"response": "Local answer", "done_reason": "stop",
                                          "prompt_eval_count": 8, "model": "local-model"}
        factory = Mock(return_value=client)
        with patch.dict(sys.modules, {"ollama": NS(Client=factory)}):
            result = generate_answer("ollama", "local-model", "prompt", 512, 2048)
        self.assertEqual(result.text, "Local answer")
        client.generate.assert_called_once_with(model="local-model", prompt="prompt",
                                                 options={"num_predict": 512, "num_ctx": 2048})

    def test_cloud_adapter_contracts_and_text_extraction(self):
        usage = NS(model_dump=lambda **kwargs: {"input_tokens": 5})
        keys = {"GEMINI_API_KEY": "g-key", "OPENAI_API_KEY": "o-key", "ANTHROPIC_API_KEY": "a-key"}
        gemini = MagicMock()
        gemini.models.generate_content.return_value = NS(
            text="Gemini answer", candidates=[NS(finish_reason=NS(value="STOP"))],
            usage_metadata=usage, model_version="gemini-version")
        gemini_manager = MagicMock()
        gemini_manager.__enter__.return_value = gemini
        genai = NS(Client=Mock(return_value=gemini_manager))
        types = NS(HttpOptions=lambda **kwargs: kwargs, GenerateContentConfig=lambda **kwargs: kwargs)
        openai = MagicMock()
        openai.responses.create.return_value = NS(output_text="OpenAI answer", status="completed",
                                                   incomplete_details=None, usage=usage, model="gpt-model")
        openai_manager = MagicMock()
        openai_manager.__enter__.return_value = openai
        anthropic = MagicMock()
        anthropic.messages.create.return_value = NS(content=[NS(type="thinking"), NS(type="text", text="Claude answer")],
                                                       stop_reason="end_turn", usage=usage, model="claude-model")
        anthropic_manager = MagicMock()
        anthropic_manager.__enter__.return_value = anthropic
        modules = {"google": NS(genai=genai), "google.genai": NS(types=types),
                   "openai": NS(OpenAI=Mock(return_value=openai_manager)),
                   "anthropic": NS(Anthropic=Mock(return_value=anthropic_manager))}
        with patch.dict(os.environ, keys), patch.dict(sys.modules, modules):
            self.assertEqual(generate_answer("gemini", "flash", "same prompt").text, "Gemini answer")
            self.assertEqual(generate_answer("openai", "gpt", "same prompt").text, "OpenAI answer")
            self.assertEqual(generate_answer("anthropic", "claude", "same prompt").text, "Claude answer")
            gemini.models.generate_content.return_value = NS(text=None, candidates=[], usage_metadata=None)
            self.assertEqual(generate_answer("gemini", "flash", "same prompt").finish_reason, "blocked_or_empty")
        self.assertEqual(gemini.models.generate_content.call_args.kwargs["contents"], "same prompt")
        self.assertFalse(openai.responses.create.call_args.kwargs["store"])
        self.assertEqual(anthropic.messages.create.call_args.kwargs["messages"], [{"role": "user", "content": "same prompt"}])

    def test_comparison_identical_prompt_failure_continuation_and_redaction(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.json"
            args = arguments("--compare", "gemini=flash", "--compare", "ollama=mini", "--results", str(path))
            with patch.dict(os.environ, {"GEMINI_API_KEY": "private-key"}), redirect_stdout(io.StringIO()), \
                    patch("model_providers.generate_answer", side_effect=[RuntimeError("429 private-key"), Answer("Evidence", "stop")]) as generate:
                self.assertEqual(run_answers("fixed prompt", args), 1)
            self.assertEqual([c.args[2] for c in generate.call_args_list], ["fixed prompt", "fixed prompt"])
            report = json.loads(path.read_text())
            self.assertEqual([r["status"] for r in report["runs"]], ["error", "ok"])
            self.assertNotIn("private-key", path.read_text())
            self.assertEqual(report["prompt_sha256"], hashlib.sha256(b"fixed prompt").hexdigest())

    def test_partial_answer_is_not_marked_successful(self):
        with redirect_stdout(io.StringIO()), patch("model_providers.generate_answer", return_value=Answer("Partial", "MAX_TOKENS")):
            self.assertEqual(run_answers("prompt", arguments()), 1)

    def test_level1_snapshot_then_swap_model_without_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            (source / "note.md").write_text("Course evidence", encoding="utf-8")
            snapshot = root / "snapshot.txt"
            argv = ["level1", "--folder", str(source), "--save-context", str(snapshot), "--prepare-only"]
            with patch.object(sys, "argv", argv), patch("level1_raw_context.run_answers") as run, redirect_stdout(io.StringIO()):
                self.assertEqual(level1_raw_context.main(), 0)
                run.assert_not_called()
            self.assertIn("Course evidence", snapshot.read_text())
            argv = ["level1", "--context-file", str(snapshot), "--provider", "gemini", "--model", "different-flash"]
            with patch.object(sys, "argv", argv), patch("level1_raw_context.extract_universal_context") as extract, \
                    patch("level1_raw_context.run_answers", return_value=0) as run:
                self.assertEqual(level1_raw_context.main(), 0)
                extract.assert_not_called()
                self.assertIn("Course evidence", run.call_args.args[0])
                self.assertEqual(run.call_args.args[1].model, "different-flash")

    def test_level2_skip_index_and_retrieve_once_for_comparison(self):
        argv = ["level2", "--skip-index", "--compare", "ollama=mini", "--compare", "gemini=flash"]
        with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()), \
                patch("level2_vector_rag.get_collection"), patch("level2_vector_rag.index_folder") as index, \
                patch("level2_vector_rag.retrieve_context", return_value="Retrieved evidence") as retrieve, \
                patch("level2_vector_rag.run_answers", return_value=0) as run:
            self.assertEqual(level2_vector_rag.main(), 0)
            index.assert_not_called()
            retrieve.assert_called_once()
            self.assertIn("Retrieved evidence", run.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
