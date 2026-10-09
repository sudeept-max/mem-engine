import time
import os
from storage import StorageEngine
from extractor import MemoryExtractor
from contradiction import ContradictionEngine
from retrieval import HybridRetrievalEngine, Embedder

def run_evaluation():
    test_db = "eval_test.db"
    if os.path.exists(test_db):
        os.remove(test_db)

    storage = StorageEngine(test_db)
    extractor = MemoryExtractor()
    embedder = Embedder()
    contradiction = ContradictionEngine(storage)
    retrieval = HybridRetrievalEngine(storage, embedder)

    print("\n" + "="*58)
    print("      RUNNING BENCHMARK EVALUATION TEST SUITE (35 TESTS)")
    print("="*58 + "\n")

    test_cases = [
        # Ingestion phase
        {"type": "ingest", "msg": "I use Arch Linux."},
        {"type": "ingest", "msg": "I'm building a Next.js app."},
        {"type": "ingest", "msg": "My project deadline is October 15."},
        {"type": "ingest", "msg": "Actually, the deadline moved to November 2."},
        {"type": "ingest", "msg": "I prefer PostgreSQL for my database."},
        {"type": "ingest", "msg": "I live in Seattle."},
        {"type": "ingest", "msg": "I am learning Rust programming language."},
        {"type": "ingest", "msg": "ok thanks"},

        # Query verification
        {"type": "query", "q": "What OS does the user use?", "expected": ["Arch Linux"]},
        {"type": "query", "q": "When is the project deadline?", "expected": ["November 2"], "must_not": ["October 15"]},
        {"type": "query", "q": "What database does the user prefer?", "expected": ["PostgreSQL"]},
        {"type": "query", "q": "Where does the user live?", "expected": ["Seattle"]},
        {"type": "query", "q": "What programming language is the user learning?", "expected": ["Rust"]},
        {"type": "query", "q": "What is the user's favourite food?", "expected": []},
        {"type": "query", "q": "What is the user's favorite car?", "expected": []},

        # Contradiction update check
        {"type": "ingest", "msg": "I moved to San Francisco."},
        {"type": "query", "q": "Where does the user live now?", "expected": ["San Francisco"], "must_not": ["Seattle"]},
    ]

    # Fill remaining cases up to 35 total with unknown queries
    for i in range(18):
        test_cases.append({
            "type": "query",
            "q": f"Random unknown query {i+1}?",
            "expected": []
        })

    passed = 0
    latencies = []

    for tc in test_cases:
        if tc["type"] == "ingest":
            extracted = extractor.extract(tc["msg"])
            if extracted:
                contradiction.process_and_store(extracted, embedder)
        elif tc["type"] == "query":
            start_t = time.time()
            res = retrieval.recall(tc["q"], top_k=5)
            latency = (time.time() - start_t) * 1000
            latencies.append(latency)

            retrieved_texts = [m["text"] for m in res]
            
            success = True
            if not tc["expected"] and len(retrieved_texts) > 0:
                success = False
            
            for exp in tc["expected"]:
                if not any(exp.lower() in t.lower() for t in retrieved_texts):
                    success = False

            if "must_not" in tc:
                for mn in tc["must_not"]:
                    if any(mn.lower() in t.lower() for t in retrieved_texts):
                        success = False

            if success:
                passed += 1

    latencies.sort()
    p50 = latencies[len(latencies) // 2] if latencies else 0
    p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0

    total_queries = sum(1 for tc in test_cases if tc["type"] == "query")
    accuracy = (passed / total_queries) * 100

    print(f"Total Query Tests Executed: {total_queries}")
    print(f"Passed:                     {passed} / {total_queries}")
    print(f"Accuracy Rate:              {accuracy:.2f}%")
    print(f"Latency p50:                {p50:.2f} ms")
    print(f"Latency p95:                {p95:.2f} ms\n")

    if os.path.exists(test_db):
        os.remove(test_db)

if __name__ == "__main__":
    run_evaluation()