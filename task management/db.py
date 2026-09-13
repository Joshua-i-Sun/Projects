# db.py
import sqlite3
from pathlib import Path

DB_PATH = Path("tasks.db")
SCHEMA_PATH = Path("schema.sql")


def get_connection() -> sqlite3.Connection:
    """Return a connection with foreign keys enforced and row access by name."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db() -> None:
    """Execute schema.sql to create tables, constraints, and indexes."""
    with get_connection() as conn:
        conn.executescript(SCHEMA_PATH.read_text())
        seed_lookup_tables(conn)


def seed_lookup_tables(conn: sqlite3.Connection) -> None:
    """Populate lookup tables with default values (idempotent)."""
    statuses = [("Pending", 1), ("In Progress", 2), ("Done", 3)]
    priorities = [("Low", 1), ("Medium", 3), ("High", 5)]
    categories = [("Work", "Professional tasks"), ("Personal", "Personal tasks")]

    conn.executemany(
        "INSERT OR IGNORE INTO statuses (name, sort_order) VALUES (?, ?)", statuses
    )
    conn.executemany(
        "INSERT OR IGNORE INTO priorities (name, level) VALUES (?, ?)", priorities
    )
    conn.executemany(
        "INSERT OR IGNORE INTO categories (name, description) VALUES (?, ?)", categories
    )
    conn.commit()