from __future__ import annotations

from typing import Any, Dict, Optional
import json
import sqlite3
from datetime import datetime, timezone
import os


def export_compliance_results(db_path: str, results: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> str:
    """Write compliance results into a SQLite database suitable for Datasette.

    Tables created if not present:
      - runs(id INTEGER PRIMARY KEY, created_at TEXT, metadata TEXT)
      - compliance_stats(run_id INTEGER, min REAL, mean REAL, max REAL, count INTEGER,
                         warning_threshold REAL, critical_threshold REAL,
                         warning_exceedances INTEGER, critical_exceedances INTEGER)
    Returns the db_path.
    """
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY,
                created_at TEXT NOT NULL,
                metadata TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS compliance_stats (
                run_id INTEGER NOT NULL,
                min REAL, mean REAL, max REAL, count INTEGER,
                warning_threshold REAL, critical_threshold REAL,
                warning_exceedances INTEGER, critical_exceedances INTEGER,
                FOREIGN KEY(run_id) REFERENCES runs(id)
            )
            """
        )
        # Insert run
        created_at = datetime.now(timezone.utc).isoformat()
        cur.execute(
            "INSERT INTO runs(created_at, metadata) VALUES (?, ?)",
            (created_at, json.dumps(metadata or {})),
        )
        run_id = cur.lastrowid
        # Insert stats
        stats = results.get("stats", {})
        thresholds = results.get("thresholds", {})
        cur.execute(
            """
            INSERT INTO compliance_stats(
                run_id, min, mean, max, count,
                warning_threshold, critical_threshold,
                warning_exceedances, critical_exceedances
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                float(stats.get("min")) if isinstance(stats.get("min"), (int, float)) else None,
                float(stats.get("mean")) if isinstance(stats.get("mean"), (int, float)) else None,
                float(stats.get("max")) if isinstance(stats.get("max"), (int, float)) else None,
                int(stats.get("count", 0)),
                float(thresholds.get("warning_celsius", 0.0)),
                float(thresholds.get("critical_celsius", 0.0)),
                int(results.get("warning_exceedances", 0)),
                int(results.get("critical_exceedances", 0)),
            ),
        )
        conn.commit()
        return db_path
    finally:
        conn.close()

