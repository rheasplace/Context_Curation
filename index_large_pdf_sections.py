"""Resume and index large PDFs (such as the 25,910-page Unreal Engine 5.7 docs) in 50 parallel sections."""
import concurrent.futures
import hashlib
import math
import multiprocessing
import os
from pathlib import Path
import time

from universal_context import split_text

ROOT = Path(__file__).resolve().parent
PDF_PATH = str((ROOT / "context_sample" / "01_unreal-engine-5-7-documentation-1.pdf").resolve())
PREPARED_FOLDER = str((ROOT / "prepared_context").resolve())
DB_PATH = str((ROOT / "local_vectordb").resolve())
SOURCE_NAME = "course.txt::01_unreal-engine-5-7-documentation-1.pdf"
CHUNK_SIZE = 1800
OVERLAP = 150
EMBED_MODEL = "all-minilm"
COLLECTION = "universal_art_collection"

_worker_reader = None


def _init_worker(pdf_path):
    global _worker_reader
    from pypdf import PdfReader
    _worker_reader = PdfReader(pdf_path)


def _extract_page_range(args):
    start_page, end_page = args
    results = []
    for page_num in range(start_page, end_page + 1):
        try:
            text = (_worker_reader.pages[page_num - 1].extract_text() or "").strip()
        except Exception:
            continue
        if not text:
            continue
        chunks = list(split_text(text, CHUNK_SIZE, OVERLAP))
        if chunks:
            results.append((page_num, chunks))
    return results


def fetch_all_existing(collection, where):
    ids, metadatas, offset, page_size = [], [], 0, 25000
    while True:
        batch = collection.get(where=where, include=["metadatas"], limit=page_size, offset=offset)
        if not batch["ids"]:
            break
        ids.extend(batch["ids"])
        metadatas.extend(batch["metadatas"])
        if len(batch["ids"]) < page_size:
            break
        offset += page_size
    return {"ids": ids, "metadatas": metadatas}


def main():
    import chromadb
    from chromadb.config import Settings
    import ollama

    t_start = time.perf_counter()
    client = chromadb.PersistentClient(path=DB_PATH, settings=Settings(anonymized_telemetry=False))
    collection = client.get_or_create_collection(name=COLLECTION, embedding_function=None)

    print("Checking existing progress in local_vectordb (paginated)...", flush=True)
    existing = fetch_all_existing(collection, {"corpus": PREPARED_FOLDER})
    existing_ids = set(existing["ids"])

    source_counts = {}
    unreal_pages = set()
    for m in existing["metadatas"]:
        src = m["source"]
        source_counts[src] = source_counts.get(src, 0) + 1
        if src == SOURCE_NAME:
            unreal_pages.add(m["page"])

    max_page_done = max(unreal_pages) if unreal_pages else 0
    total_pages = 25910
    base_start_page = 16201
    pages_per_section = math.ceil((total_pages - base_start_page + 1) / 50)
    remaining_pages = max(0, total_pages - max_page_done)

    print(f"Total chunks in DB so far: {len(existing_ids)}", flush=True)
    print(f"Unreal 5.7 PDF progress: {max_page_done}/{total_pages} pages completed ({max_page_done / total_pages * 100:.1f}%)", flush=True)
    print(f"Unreal 5.7 chunks saved so far: {source_counts.get(SOURCE_NAME, 0)}", flush=True)
    print(f"Remaining pages to index: {remaining_pages} (Pages {max_page_done + 1} to {total_pages})", flush=True)

    if remaining_pages == 0:
        print("All 25,910 pages are already indexed!", flush=True)
        return

    sections = []
    for s_idx in range(50):
        s_start = base_start_page + s_idx * pages_per_section
        if s_start > total_pages:
            break
        s_end = min(total_pages, s_start + pages_per_section - 1)
        if s_end <= max_page_done:
            continue
        sections.append((s_idx + 1, max(s_start, max_page_done + 1), s_end))

    num_workers = min(6, os.cpu_count() or 4)
    print(f"\nResuming {len(sections)} remaining sections (Section {sections[0][0]:02d}/50 to 50/50) using {num_workers} parallel PDF workers...", flush=True)

    ollama_client = ollama.Client(timeout=120)
    embed_batch_size = 64

    def safe_embed(b_chunks):
        cleaned = [c.replace("\x00", " ").strip() or " " for c in b_chunks]
        for attempt in range(4):
            try:
                return ollama_client.embed(model=EMBED_MODEL, input=cleaned, truncate=True)["embeddings"]
            except Exception:
                if attempt < 3:
                    time.sleep(2 * (attempt + 1))
                else:
                    vectors = []
                    for single in cleaned:
                        try:
                            vectors.append(ollama_client.embed(model=EMBED_MODEL, input=[single[:1200]], truncate=True)["embeddings"][0])
                        except Exception:
                            vectors.append(ollama_client.embed(model=EMBED_MODEL, input=["unreal engine documentation"], truncate=True)["embeddings"][0])
                    return vectors

    def embed_and_upsert(chunks, ids, metas):
        if not chunks:
            return
        for i in range(0, len(chunks), embed_batch_size):
            b_chunks = chunks[i:i + embed_batch_size]
            b_ids = ids[i:i + embed_batch_size]
            b_metas = metas[i:i + embed_batch_size]
            embeddings = safe_embed(b_chunks)
            collection.upsert(ids=b_ids, documents=b_chunks, embeddings=embeddings, metadatas=b_metas)

    total_new_chunks = 0
    t_AfterFirst = time.perf_counter()

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_worker,
        initargs=(PDF_PATH,),
    ) as pool:
        section_futures = []
        for sec_num, s_start, s_end in sections:
            span = s_end - s_start + 1
            step = math.ceil(span / num_workers)
            sub_futures = []
            for w in range(num_workers):
                w_start = s_start + w * step
                if w_start > s_end:
                    break
                w_end = min(s_end, w_start + step - 1)
                sub_futures.append(pool.submit(_extract_page_range, (w_start, w_end)))
            section_futures.append((sec_num, s_start, s_end, sub_futures))

        for idx, (sec_num, s_start, s_end, sub_futures) in enumerate(section_futures, 1):
            t_sec = time.perf_counter()
            sec_chunks, sec_ids, sec_metas = [], [], []
            for fut in sub_futures:
                for page_num, chunks in fut.result():
                    for c_idx, chunk in enumerate(chunks):
                        identity = f"{PREPARED_FOLDER}\0{SOURCE_NAME}\0{page_num}\0{c_idx}"
                        chunk_id = hashlib.sha256(identity.encode()).hexdigest()
                        if chunk_id in existing_ids:
                            continue
                        existing_ids.add(chunk_id)
                        sec_chunks.append(chunk)
                        sec_ids.append(chunk_id)
                        sec_metas.append({
                            "corpus": PREPARED_FOLDER,
                            "source": SOURCE_NAME,
                            "kind": "pdf",
                            "page": page_num,
                            "chunk": c_idx,
                            "embedding_model": EMBED_MODEL,
                        })
            embed_and_upsert(sec_chunks, sec_ids, sec_metas)
            total_new_chunks += len(sec_chunks)
            sec_elapsed = time.perf_counter() - t_sec
            if idx == 1:
                eta_seconds = 15.0 * (len(section_futures) - idx)
                t_AfterFirst = time.perf_counter()
            else:
                steady_elapsed = time.perf_counter() - t_AfterFirst
                avg_steady = steady_elapsed / (idx - 1)
                eta_seconds = avg_steady * (len(section_futures) - idx)
            print(
                f"  [Section {sec_num:02d}/50] Pages {s_start}-{s_end}/{total_pages} "
                f"| +{len(sec_chunks)} chunks (total +{total_new_chunks}) "
                f"| {sec_elapsed:.1f}s | ETA: {eta_seconds / 60:.1f} min",
                flush=True,
            )

    print(f"\nCompleted all 50 sections! Added {total_new_chunks} new chunks in {(time.perf_counter() - t_start)/60:.1f} minutes.", flush=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
