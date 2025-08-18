import sqlite3
from datetime import datetime

DB_NAME = "tasks.db"


def init_db():
    """Initialize database and create tasks table if not exists."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT,
            due_date TEXT,
            status TEXT DEFAULT 'Pending'
        )
        """
    )
    conn.commit()
    conn.close()


def add_task(title, description, due_date):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO tasks (title, description, due_date) VALUES (?, ?, ?)",
        (title, description, due_date),
    )
    conn.commit()
    conn.close()
    print("Task added successfully")


def view_tasks():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM tasks")
    tasks = cursor.fetchall()
    conn.close()

    if not tasks:
        print("No tasks found.")
    else:
        print("\n=== Your Tasks ===")
        for task in tasks:
            print(
                f"[{task[0]}] {task[1]} (Due: {task[3]}) - {task[4]} \n   {task[2]}"
            )


def update_task(task_id, new_status):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("UPDATE tasks SET status=? WHERE id=?", (new_status, task_id))
    conn.commit()
    conn.close()
    print("Task updated successfully")


def delete_task(task_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM tasks WHERE id=?", (task_id,))
    conn.commit()
    conn.close()
    print("🗑️ Task deleted successfully")


def main():
    init_db()

    while True:
        print("\n=== Task Manager ===")
        print("1. Add Task")
        print("2. View Tasks")
        print("3. Update Task Status")
        print("4. Delete Task")
        print("5. Exit")

        choice = input("Choose an option (1-5): ")

        if choice == "1":
            title = input("Enter task title: ")
            description = input("Enter description: ")
            due_date = input("Enter due date (YYYY-MM-DD): ")
            add_task(title, description, due_date)

        elif choice == "2":
            view_tasks()

        elif choice == "3":
            task_id = int(input("Enter task ID to update: "))
            new_status = input("Enter new status (Pending / In Progress / Done): ")
            update_task(task_id, new_status)

        elif choice == "4":
            task_id = int(input("Enter task ID to delete: "))
            delete_task(task_id)

        elif choice == "5":
            print("Goodbye")
            break

        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    main()
