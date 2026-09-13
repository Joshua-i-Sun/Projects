-- schema.sql

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS users (
    user_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    username    TEXT    NOT NULL UNIQUE,
    email       TEXT    NOT NULL UNIQUE CHECK (email LIKE '%_@_%._%'),
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS categories (
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE,
    description TEXT
);

CREATE TABLE IF NOT EXISTS statuses (
    status_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE,
    sort_order  INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS priorities (
    priority_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE,
    level       INTEGER NOT NULL UNIQUE CHECK (level BETWEEN 1 AND 5)
);

CREATE TABLE IF NOT EXISTS tasks (
    task_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      INTEGER NOT NULL,
    category_id  INTEGER,
    status_id    INTEGER NOT NULL,
    priority_id  INTEGER NOT NULL,
    title        TEXT    NOT NULL CHECK (length(title) > 0),
    description  TEXT,
    due_date     TEXT    CHECK (due_date IS NULL OR due_date LIKE '____-__-__'),
    created_at   TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at   TEXT    NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (user_id)     REFERENCES users(user_id)     ON DELETE CASCADE,
    FOREIGN KEY (category_id) REFERENCES categories(category_id) ON DELETE SET NULL,
    FOREIGN KEY (status_id)   REFERENCES statuses(status_id)     ON DELETE RESTRICT,
    FOREIGN KEY (priority_id) REFERENCES priorities(priority_id) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS tags (
    tag_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name   TEXT    NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS task_tags (
    task_id INTEGER NOT NULL,
    tag_id  INTEGER NOT NULL,
    PRIMARY KEY (task_id, tag_id),
    FOREIGN KEY (task_id) REFERENCES tasks(task_id) ON DELETE CASCADE,
    FOREIGN KEY (tag_id)  REFERENCES tags(tag_id)   ON DELETE CASCADE
);

-- Indexes to optimize common query patterns
CREATE INDEX IF NOT EXISTS idx_tasks_user_id     ON tasks(user_id);
CREATE INDEX IF NOT EXISTS idx_tasks_status_id   ON tasks(status_id);
CREATE INDEX IF NOT EXISTS idx_tasks_due_date    ON tasks(due_date);
CREATE INDEX IF NOT EXISTS idx_tasks_priority_id ON tasks(priority_id);
CREATE INDEX IF NOT EXISTS idx_task_tags_tag_id  ON task_tags(tag_id);