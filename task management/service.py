# service.py
from typing import Optional, List
import repository as repo
from db import get_connection


def register_user(username: str, email: str) -> int:
    existing = repo.get_user_by_username(username)
    if existing:
        raise ValueError(f"Username '{username}' is already taken.")
    return repo.create_user(username, email)


def add_task(user_id, title, description, due_date, category_name, priority_name):
    """Resolve lookup names to IDs, then delegate to the repository."""
    with get_connection() as conn:
        status_id = conn.execute(
            "SELECT status_id FROM statuses WHERE name = 'Pending'"
        ).fetchone()["status_id"]

        priority_row = conn.execute(
            "SELECT priority_id FROM priorities WHERE name = ?", (priority_name,)
        ).fetchone()
        if not priority_row:
            raise ValueError(f"Unknown priority: {priority_name}")

        category_row = conn.execute(
            "SELECT category_id FROM categories WHERE name = ?", (category_name,)
        ).fetchone()
        category_id = category_row["category_id"] if category_row else None

    return repo.create_task(
        user_id=user_id,
        title=title,
        description=description,
        due_date=due_date,
        category_id=category_id,
        status_id=status_id,
        priority_id=priority_row["priority_id"],
    )


def mark_task_done(task_id: int) -> None:
    with get_connection() as conn:
        done_id = conn.execute(
            "SELECT status_id FROM statuses WHERE name = 'Done'"
        ).fetchone()["status_id"]
    repo.update_task_status(task_id, done_id)


def get_user_tasks(user_id: int) -> List:
    return repo.list_tasks_for_user(user_id)