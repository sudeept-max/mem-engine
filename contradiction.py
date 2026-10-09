import math
from typing import List, Dict, Any
from config import Config

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    return dot / (n1 * n2) if n1 > 0 and n2 > 0 else 0.0

class ContradictionEngine:
    def __init__(self, storage):
        self.storage = storage

    def process_and_store(self, extracted_memories: List[Dict[str, Any]], embedder) -> List[Dict[str, Any]]:
        stored_results = []
        active_memories = self.storage.get_active_memories()

        for new_mem in extracted_memories:
            new_emb = embedder.encode(new_mem["text"])
            topic_key = new_mem.get("topic_key")

            replaced_ids = []
            for existing in active_memories:
                sim = cosine_similarity(new_emb, existing["embedding"])
                same_topic = (topic_key and existing.get("topic_key") == topic_key and topic_key != "general")
                high_sim = sim >= Config.CONTRADICTION_SIM_THRESHOLD

                if same_topic or high_sim:
                    replaced_ids.append(existing["id"])

            new_id = self.storage.insert_memory(new_mem, new_emb)
            new_mem["id"] = new_id
            new_mem["status"] = "ACTIVE"

            for old_id in replaced_ids:
                self.storage.mark_replaced(old_id, new_id)

            stored_results.append(new_mem)

        return stored_results