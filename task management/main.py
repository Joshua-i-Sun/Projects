# main.py
from db import init_db, get_connection
import service
import repository as repo


def prompt(msg: str) -> str:
    return input(msg).strip()


def main():
    init_db()
    print("=== Task Management & Data Tracking System ===")

    username = prompt("Enter username: ")
    user = repo.get_user_by_username(username)
    if not user:
        email = prompt("Enter email to register: ")
        user_id = service.register_user(username, email)
        print(f"Registered new user (id={user_id}).")
    else:
        user_id = user["user_id"]
        print(f"Welcome back, {username}.")

    while True:
        print("\n1. Add Task  2. View My Tasks  3. Mark Done  4. Delete  5. Query Plan  6. Exit")
        choice = prompt("Choose (1-6): ")

        if choice == "1":
            title = prompt("Title: ")
            desc = prompt("Description: ")
            due = prompt("Due date (YYYY-MM-DD): ")
            cat = prompt("Category (Work/Personal): ")
            pri = prompt("Priority (Low/Medium/High): ")
            tid = service.add_task(user_id, title, desc, due, cat, pri)
            print(f"Task created (id={tid}).")

        elif choice == "2":
            rows = service.get_user_tasks(user_id)
            if not rows:
                print("No tasks found.")
            for r in rows:
                print(f"[{r['task_id']}] {r['title']} | {r['priority']} | {r['status']} | due {r['due_date']}")

        elif choice == "3":
            tid = int(prompt("Task ID to mark done: "))
            service.mark_task_done(tid)
            print("Marked as Done.")

        elif choice == "4":
            tid = int(prompt("Task ID to delete: "))
            repo.delete_task(tid)
            print("Deleted.")

        elif choice == "5":
            sql = """
                SELECT t.task_id, t.title
                FROM tasks t
                WHERE t.user_id = ? AND t.status_id = ?
                ORDER BY t.due_date
            """
            with get_connection() as conn:
                uid = user_id
                sid = conn.execute("SELECT status_id FROM statuses WHERE name='Pending'").fetchone()["status_id"]
            plan = repo.explain_plan(sql, (uid, sid))
            print("\nEXPLAIN QUERY PLAN:")
            for step in plan:
                print(f"  {step}")

        elif choice == "6":
            print("Goodbye.")
            break


if __name__ == "__main__":
    main()