"""
Universal Zero-Storage Registry for TorchJIT & UPAAS
Role: Database strictly keeps CIDs (hashes), chunk indices, and decryption keys.
ZERO tensor bytes are ever saved in the DB.
All weights live permanently on the global IPFS distributed network.
"""
import os
import sqlite3
from typing import Optional, List, Dict, Any

class StatePooler:
    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or os.environ.get("DATABASE_URL", "")
        self.is_remote = bool(self.database_url and self.database_url.startswith("postgres"))
        self._pg_pool = None
        self.local_db_path = os.path.expanduser("~/.tjit/registry.db")

        if self.is_remote:
            self._init_remote_pool()
        else:
            self._init_local_db()

    def _init_local_db(self):
        os.makedirs(os.path.dirname(self.local_db_path), exist_ok=True)
        with sqlite3.connect(self.local_db_path) as conn:
            cur = conn.cursor()
            cur.execute("""
            CREATE TABLE IF NOT EXISTS model_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                layer_index INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                cid TEXT NOT NULL,
                decryption_key TEXT,
                byte_size INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(model_id, layer_index, chunk_index)
            );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_local_model_layer ON model_chunks(model_id, layer_index);")
            conn.commit()

    def _init_remote_pool(self):
        try:
            import psycopg2
            from psycopg2 import pool
            self._pg_pool = pool.SimpleConnectionPool(
                minconn=1,
                maxconn=5,
                dsn=self.database_url
            )
        except Exception as e:
            print(f"[TorchJIT Pooler] Remote pooler failed: {e}. Falling back to local offline DB.")
            self.is_remote = False
            self._init_local_db()

    def register_chunk_hash(self, model_id: str, layer_idx: int, chunk_idx: int, cid: str, byte_size: int, dec_key: str = None):
        """Save ONLY the IPFS CID hash and decryption key into the database."""
        if self.is_remote and self._pg_pool:
            conn = self._pg_pool.getconn()
            try:
                with conn.cursor() as cur:
                    query = """
                    INSERT INTO model_chunks (model_id, layer_index, chunk_index, cid, decryption_key, byte_size)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (model_id, layer_index, chunk_index) 
                    DO UPDATE SET cid = EXCLUDED.cid, decryption_key = EXCLUDED.decryption_key, byte_size = EXCLUDED.byte_size;
                    """
                    cur.execute(query, (model_id, layer_idx, chunk_idx, cid, dec_key, byte_size))
                conn.commit()
            finally:
                self._pg_pool.putconn(conn)
        else:
            with sqlite3.connect(self.local_db_path) as conn:
                cur = conn.cursor()
                cur.execute("""
                INSERT INTO model_chunks (model_id, layer_index, chunk_index, cid, decryption_key, byte_size)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(model_id, layer_index, chunk_index)
                DO UPDATE SET cid=excluded.cid, decryption_key=excluded.decryption_key, byte_size=excluded.byte_size;
                """, (model_id, layer_idx, chunk_idx, cid, dec_key, byte_size))
                conn.commit()

    def get_layer_manifest(self, model_id: str, layer_idx: int) -> List[Dict[str, Any]]:
        """Fetch the list of IPFS CIDs and decryption keys needed to stream a layer."""
        if self.is_remote and self._pg_pool:
            from psycopg2.extras import RealDictCursor
            conn = self._pg_pool.getconn()
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                    SELECT chunk_index, cid, decryption_key, byte_size FROM model_chunks
                    WHERE model_id = %s AND layer_index = %s ORDER BY chunk_index ASC;
                    """, (model_id, layer_idx))
                    return [dict(r) for r in cur.fetchall()]
            finally:
                self._pg_pool.putconn(conn)
        else:
            with sqlite3.connect(self.local_db_path) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()
                cur.execute("""
                SELECT chunk_index, cid, decryption_key, byte_size FROM model_chunks
                WHERE model_id = ? AND layer_index = ? ORDER BY chunk_index ASC;
                """, (model_id, layer_idx))
                return [dict(r) for r in cur.fetchall()]

# Aliases
StatePooler.register_cid_chunk = StatePooler.register_chunk_hash
StatePooler.get_layer_chunks = StatePooler.get_layer_manifest
TransactionPooler = StatePooler
