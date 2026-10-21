"""Index universal context and answer using semantic retrieval."""
import argparse
import hashlib
from pathlib import Path
from universal_context import load_corpus_documents as load_documents, split_text
from model_providers import add_model_arguments, validate_model_arguments, build_prompt, run_answers

COLLECTION = "universal_art_collection"
EMBED_MODEL = "all-minilm"


def get_collection(db_path="./local_vectordb"):
    import chromadb
    return chromadb.PersistentClient(path=db_path).get_or_create_collection(
        name=COLLECTION, embedding_function=None,
    )


def index_folder(collection, folder, chunk_size=600, overlap=80, batch_size=64):
    import ollama
    corpus = str(Path(folder).resolve())
    documents = load_documents(folder)
    
    # 1. Check existing IDs in ChromaDB to avoid re-embedding unchanged chunks
    existing_records = collection.get(where={"corpus": corpus}, include=[])
    existing_ids = set(existing_records["ids"]) if existing_records else set()
    
    current_ids = set()
    pending_chunks = []
    pending_ids = []
    pending_metas = []

    # 2. Slice documents and queue only new/modified chunks
    for document in documents:
        for index, chunk in enumerate(split_text(document.text, chunk_size, overlap)):
            identity = f"{corpus}\0{document.source}\0{document.page}\0{index}"
            chunk_id = hashlib.sha256(identity.encode()).hexdigest()
            current_ids.add(chunk_id)

            if chunk_id not in existing_ids:
                pending_chunks.append(chunk)
                pending_ids.append(chunk_id)
                pending_metas.append({
                    "corpus": corpus, "source": document.source,
                    "kind": document.kind, "page": document.page,
                    "chunk": index, "embedding_model": EMBED_MODEL
                })

    # 3. Batch process new embeddings to prevent thousands of single API calls
    if pending_chunks:
        print(f"Indexing {len(pending_chunks)} new/updated chunks...")
        for i in range(0, len(pending_chunks), batch_size):
            batch_texts = pending_chunks[i:i + batch_size]
            batch_ids = pending_ids[i:i + batch_size]
            batch_metas = pending_metas[i:i + batch_size]

            # Send batch to Ollama for parallel GPU embedding
            resp = ollama.embed(model=EMBED_MODEL, input=batch_texts, truncate=True)
            batch_embeddings = resp["embeddings"]

            collection.upsert(
                ids=batch_ids,
                documents=batch_texts,
                embeddings=batch_embeddings,
                metadatas=batch_metas,
            )
    else:
        print(f"All {len(current_ids)} chunks are up to date.")

    # 4. Remove stale chunks
    stale = sorted(existing_ids - current_ids)
    for start in range(0, len(stale), 100):
        collection.delete(ids=stale[start:start + 100])

    return len(current_ids)


def retrieve_context(collection, query, folder="prepared_context", top_k=3):
    import ollama
    if top_k < 1:
        raise ValueError("top_k must be positive")
    corpus = str(Path(folder).resolve())
    count = len(collection.get(where={"corpus": corpus}, include=[])["ids"])
    if not count:
        return ""
    embeddings = ollama.embed(model=EMBED_MODEL, input=query, truncate=False)["embeddings"]
    results = collection.query(query_embeddings=embeddings, n_results=min(top_k, count),
                               where={"corpus": corpus}, include=["documents", "metadatas"])
    sections = []
    for text, meta in zip(results["documents"][0], results["metadatas"][0]):
        page = f", page {meta['page']}" if meta["page"] else ""
        sections.append(f"--- {meta['source']}{page} ({meta['kind']}, chunk {meta['chunk']}) ---\n{text}")
    return "\n\n".join(sections)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", default="prepared_context")
    parser.add_argument("--db", default="./local_vectordb")
    parser.add_argument("--query", default="What are the key themes and how are they related? Cite the sources.")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=600)
    parser.add_argument("--overlap", type=int, default=80)
    parser.add_argument("--skip-index", action="store_true", help="Query the existing index without reprocessing files")
    parser.add_argument("--index-only", action="store_true", help="Prepare Level 5 without making an answer-generation call")
    add_model_arguments(parser, default_provider="gemini")
    args = parser.parse_args()
    validate_model_arguments(parser, args)
    if args.skip_index and args.index_only:
        parser.error("--index-only cannot be combined with --skip-index")
    if not 0 <= args.overlap < args.chunk_size or args.top_k < 1:
        parser.error("Require 0 <= overlap < chunk-size and top-k >= 1")
    collection = get_collection(args.db)
    
    # Auto-bypass re-indexing if chunks exist and --folder was not forced
    corpus = str(Path(args.folder).resolve())
    existing = collection.get(where={"corpus": corpus}, include=[])
    has_existing = bool(existing and existing["ids"])

    if not args.skip_index and not has_existing:
        count = index_folder(collection, args.folder, args.chunk_size, args.overlap)
        print(f"Indexed {count} chunks.")
        if args.index_only:
            return 0
    elif has_existing and not args.skip_index:
        print(f"Database already populated ({len(existing['ids'])} chunks). Skipping re-index.")

    context = retrieve_context(collection, args.query, args.folder, args.top_k)
    if not context:
        print("No indexed context found. Run without --skip-index for this folder first.")
        return 1
    print(context)
    return run_answers(build_prompt(context, args.query), args)


if __name__ == "__main__":
    raise SystemExit(main())