import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from config import Config

class StorageEngine:
    def __init__(self, db_path: str = Config.DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    type TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'ACTIVE',
                    replaced_by INTEGER NULL,
                    importance_score REAL DEFAULT 3.0,
                    topic_key TEXT NULL,
                    embedding TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    source_message TEXT NOT NULL,
                    FOREIGN KEY(replaced_by) REFERENCES memories(id)
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON memories(status);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_topic ON memories(topic_key);")
            conn.commit()

    def insert_memory(self, memory: Dict[str, Any], embedding: List[float]) -> int:
        timestamp = datetime.utcnow().isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO memories (text, type, status, importance_score, topic_key, embedding, timestamp, source_message)
                VALUES (?, ?, 'ACTIVE', ?, ?, ?, ?, ?)
            """, (
                memory["text"],
                memory["type"],
                memory.get("importance_score", 3.0),
                memory.get("topic_key"),
                json.dumps(embedding),
                timestamp,
                memory["source_message"]
            ))
            conn.commit()
            return cursor.lastrowid

    def mark_replaced(self, old_id: int, new_id: int):
        with self._get_connection() as conn:
            conn.execute("""
                UPDATE memories
                SET status = 'REPLACED', replaced_by = ?
                WHERE id = ?
            """, (new_id, old_id))
            conn.commit()

    def get_active_memories(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM memories WHERE status = 'ACTIVE'").fetchall()
            return [self._row_to_dict(r) for r in rows]

    def get_all_memories(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            if status:
                rows = conn.execute("SELECT * FROM memories WHERE status = ?", (status.upper(),)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM memories").fetchall()
            return [self._row_to_dict(r) for r in rows]

    def delete_memory(self, memory_id: int) -> bool:
        with self._get_connection() as conn:
            cursor = conn.execute("UPDATE memories SET status = 'DELETED' WHERE id = ?", (memory_id,))
            conn.commit()
            return cursor.rowcount > 0

    def _row_to_dict(self, row) -> Dict[str, Any]:
        d = dict(row)
        d["embedding"] = json.loads(d["embedding"])
        return d