"""Answer from the persisted Gemini-extracted graph of prepared context."""
import argparse
from graph_context import get_graph_context
from model_providers import add_model_arguments, validate_model_arguments, build_prompt, run_answers

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="What are the key themes and how are they related? Cite the sources.")
    parser.add_argument("--entity", default="", help="Optional entity filter; otherwise rank relationships by the question")
    parser.add_argument("--folder", default="prepared_context")
    parser.add_argument("--graph", default="local_graph.json")
    add_model_arguments(parser, default_provider="gemini")
    args = parser.parse_args()
    validate_model_arguments(parser, args)

    entity = args.entity
    try:
        graph_context = get_graph_context(args.graph, args.folder, entity, args.query)
    except ValueError as exc:
        print(exc)
        return 1
    print(f"--- Retrieved Graph Logic: {graph_context} ---")

    # Generate the final answer
    return run_answers(build_prompt(f"Source-backed extracted graph: {graph_context}", args.query), args)


if __name__ == "__main__":
    raise SystemExit(main())
