"""Run the recursive context-stuffing baseline."""
import argparse
from pathlib import Path
from universal_context import extract_universal_context
from model_providers import add_model_arguments, validate_model_arguments, build_prompt, run_answers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--folder", default="context_sample")
    source.add_argument("--context-file", type=Path, help="Reuse a prepared UTF-8 context snapshot")
    parser.add_argument("--save-context", type=Path, help="Save extracted text outside the input folder")
    parser.add_argument("--prepare-only", action="store_true", help="Save context without calling an answer model")
    parser.add_argument("--query", default="Summarize the key themes in these documents and images.")
    add_model_arguments(parser)
    args = parser.parse_args()
    validate_model_arguments(parser, args)
    if args.prepare_only and not args.save_context:
        parser.error("--prepare-only requires --save-context")
    if args.save_context and not args.context_file and args.save_context.resolve().is_relative_to(Path(args.folder).resolve()):
        parser.error("Save the snapshot outside --folder to avoid indexing it as a source")
    context = (args.context_file.read_text(encoding="utf-8") if args.context_file
               else extract_universal_context(args.folder))
    if not context:
        print("No supported, nonempty context found.")
        return 1
    if args.save_context:
        args.save_context.parent.mkdir(parents=True, exist_ok=True)
        args.save_context.write_text(context, encoding="utf-8")
        print(f"Saved context: {args.save_context}")
    if args.prepare_only:
        return 0
    return run_answers(build_prompt(context, args.query), args)


if __name__ == "__main__":
    raise SystemExit(main())
