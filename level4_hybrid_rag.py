"""Combine vector evidence and Gemini-extracted graph relationships from prepared context."""
import argparse
from graph_context import get_graph_context
from model_providers import add_model_arguments, validate_model_arguments, build_prompt, run_answers, positive_int

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="What are the key themes and how are they related? Cite the sources.")
    parser.add_argument("--entity", default="", help="Optional entity filter; otherwise rank relationships by the question")
    parser.add_argument("--graph", default="local_graph.json")
    parser.add_argument("--folder", default="prepared_context")
    parser.add_argument("--db", default="./local_vectordb")
    parser.add_argument("--top-k", type=positive_int, default=3)
    add_model_arguments(parser, default_provider="gemini")
    args = parser.parse_args()
    validate_model_arguments(parser, args)
    from level2_vector_rag import get_collection, retrieve_context

    # 1. Setup Vector DB (Assuming it's populated from Level 2)
    collection = get_collection(args.db)

    # 3. Hybrid Execution
    user_query = args.query

    # A) Retrieve semantic evidence from the same prepared corpus.
    vector_context = retrieve_context(collection, user_query, args.folder, args.top_k)
    if not vector_context:
        print("Run level2_vector_rag.py with the same --folder and --db first.")
        return 1

    # B) Retrieve persisted relationships by entity or question terms.
    extracted_entity = args.entity
    try:
        graph_context = get_graph_context(args.graph, args.folder, extracted_entity, user_query)
    except ValueError as exc:
        print(exc)
        return 1

    print(f"--- Vector Context: {vector_context} ---")
    print(f"--- Graph Context: {graph_context} ---")

    # C) Combined Reasoning
    hybrid_context = f"Text Reference: {vector_context}\nSource-backed extracted graph: {graph_context or 'No relationships found.'}"
    return run_answers(build_prompt(hybrid_context, user_query), args)


if __name__ == "__main__":
    raise SystemExit(main())
