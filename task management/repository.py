# repository.py
import sqlite3
from typing import Optional, List, Dict, Any
from db import get_connection


# ---------- USERS ----------

def create_user(username: str, email: str) -> int:
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO users (username, email) VALUES (?, ?)", (username, email)
        )
        conn.commit()
        return cur.lastrowid


def get_user_by_username(username: str) -> Optional[sqlite3.Row]:
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()


# ---------- TASKS (CRUD) ----------

def create_task(
    user_id: int,
    title: str,
    description: str,
    due_date: str,
    category_id: int,
    status_id: int,
    priority_id: int,
) -> int:
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO tasks
                (user_id, title, description, due_date, category_id, status_id, priority_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (user_id, title, description, due_date, category_id, status_id, priority_id),
        )
        conn.commit()
        return cur.lastrowid


def get_task(task_id: int) -> Optional[sqlite3.Row]:
    """Fetch a single task with its joined lookup values."""
    with get_connection() as conn:
        return conn.execute(
            """
            SELECT t.task_id, t.title, t.description, t.due_date,
                   u.username, c.name AS category,
                   s.name AS status, p.name AS priority
            FROM tasks t
            JOIN users      u ON u.user_id     = t.user_id
            JOIN statuses   s ON s.status_id   = t.status_id
            JOIN priorities p ON p.priority_id = t.priority_id
            LEFT JOIN categories c ON c.category_id = t.category_id
            WHERE t.task_id = ?
            """,
            (task_id,),
        ).fetchone()


def list_tasks_for_user(user_id: int) -> List[sqlite3.Row]:
    with get_connection() as conn:
        return conn.execute(
            """
            SELECT t.task_id, t.title, t.due_date,
                   s.name AS status, p.name AS priority
            FROM tasks t
            JOIN statuses   s ON s.status_id   = t.status_id
            JOIN priorities p ON p.priority_id = t.priority_id
            WHERE t.user_id = ?
            ORDER BY p.level DESC, t.due_date ASC
            """,
            (user_id,),
        ).fetchall()


def update_task_status(task_id: int, status_id: int) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE tasks SET status_id = ?, updated_at = datetime('now') WHERE task_id = ?",
            (status_id, task_id),
        )
        conn.commit()


def delete_task(task_id: int) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM tasks WHERE task_id = ?", (task_id,))
        conn.commit()


# ---------- TAGS (many-to-many) ----------

def add_tag_to_task(task_id: int, tag_name: str) -> None:
    with get_connection() as conn:
        conn.execute("INSERT OR IGNORE INTO tags (name) VALUES (?)", (tag_name,))
        tag_id = conn.execute(
            "SELECT tag_id FROM tags WHERE name = ?", (tag_name,)
        ).fetchone()["tag_id"]
        conn.execute(
            "INSERT OR IGNORE INTO task_tags (task_id, tag_id) VALUES (?, ?)",
            (task_id, tag_id),
        )
        conn.commit()


# ---------- QUERY OPTIMIZATION DEMO ----------

def explain_plan(sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
    """Run EXPLAIN QUERY PLAN and return the execution steps."""
    with get_connection() as conn:
        rows = conn.execute(f"EXPLAIN QUERY PLAN {sql}", params).fetchall()
        return [dict(r) for r in rows]