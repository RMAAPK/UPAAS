"""
Swarm Database Exporter & Local Sync Tool
Users can sync the decentralized model manifest directly to local SQLite:
  python -m upaas.export_db --pull
  python -m upaas.export_db --push
"""
import os
import sys
import sqlite3
import argparse
import urllib.request
import json

DEFAULT_LOCAL_PATH = os.path.expanduser("~/.tjit/swarm_manifest.db")
BUILTIN_DB_PATH = os.path.join(os.path.dirname(__file__), "swarm_manifest.db")

def pull_local(target_path: str = DEFAULT_LOCAL_PATH, source_path: str = BUILTIN_DB_PATH):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with open(source_path, "rb") as src, open(target_path, "wb") as dst:
        dst.write(src.read())
    print(f"📦 Pre-made decentralized swarm manifest synced locally to: {target_path}")
    print(f"🔗 Connect directly via: sqlite:///{target_path}")

def dump_json(source_path: str = BUILTIN_DB_PATH) -> dict:
    conn = sqlite3.connect(source_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM models")
    models = [dict(r) for r in cur.fetchall()]
    cur.execute("SELECT * FROM model_chunks")
    chunks = [dict(r) for r in cur.fetchall()]
    return {"models": models, "chunks": chunks}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UPAAS Swarm DB Sync")
    parser.add_argument("--pull", action="store_true", help="Pull pre-made DB to ~/.tjit/swarm_manifest.db")
    parser.add_argument("--export-json", action="store_true", help="Dump manifest as JSON")
    args = parser.parse_args()

    if args.export_json:
        print(json.dumps(dump_json(), indent=2))
    else:
        pull_local()
