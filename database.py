"""
database.py — SQLite persistence layer for Farmer Advisory System
"""
import sqlite3
import os
from datetime import datetime

DB_PATH = "farmer_advisory.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            query_text  TEXT,
            image_path  TEXT,
            disease     TEXT,
            confidence  REAL,
            timestamp   TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_prediction(image_path, disease, confidence, query_text=None):
    conn = get_connection()
    conn.execute(
        "INSERT INTO predictions (query_text, image_path, disease, confidence, timestamp) VALUES (?,?,?,?,?)",
        (query_text, image_path, disease, confidence,
         datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    conn.commit()
    conn.close()


def get_recent_history(limit=6):
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_history():
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM predictions ORDER BY id DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def clear_all_history():
    conn = get_connection()
    conn.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()


def get_stats():
    conn = get_connection()
    total  = conn.execute("SELECT COUNT(*) FROM predictions WHERE image_path IS NOT NULL").fetchone()[0]
    sick   = conn.execute("SELECT COUNT(*) FROM predictions WHERE disease NOT LIKE '%healthy%' AND disease NOT LIKE '%Query%' AND image_path IS NOT NULL").fetchone()[0]
    healthy = conn.execute("SELECT COUNT(*) FROM predictions WHERE disease LIKE '%healthy%'").fetchone()[0]
    queries = conn.execute("SELECT COUNT(*) FROM predictions WHERE image_path IS NULL").fetchone()[0]
    conn.close()
    return {"total_scans": total, "diseases_detected": sick,
            "healthy_plants": healthy, "total_queries": queries}
