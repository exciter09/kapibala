"""Small SQLite-backed queue for delayed user messages."""

import sqlite3
from typing import Literal, TypedDict


class QueuedMessage(TypedDict):
    id: int
    message: str
    last_agent_message: str | None
    fixed_reply: str | None


class EscalatedMessage(QueuedMessage):
    created_at: float


class HumanMessage(TypedDict):
    id: int
    sender: Literal["user", "admin"]
    message: str
    created_at: float


class ConversationState(TypedDict):
    status: str
    signal_streak: int


class MessageQueue:
    def __init__(self, path: str) -> None:
        self.path = path
        with self._connect() as db:
            db.execute(
                """CREATE TABLE IF NOT EXISTS pending_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    message TEXT NOT NULL,
                    last_agent_message TEXT,
                    ready_at REAL NOT NULL,
                    fixed_reply TEXT
                )"""
            )
            columns = {
                row[1] for row in db.execute("PRAGMA table_info(pending_messages)")
            }
            if "fixed_reply" not in columns:
                db.execute("ALTER TABLE pending_messages ADD COLUMN fixed_reply TEXT")
            db.execute(
                """CREATE TABLE IF NOT EXISTS conversation_states (
                    conversation_id TEXT PRIMARY KEY,
                    customer_id TEXT,
                    status TEXT NOT NULL,
                    signal_streak INTEGER NOT NULL DEFAULT 0
                )"""
            )
            state_columns = {
                row[1] for row in db.execute("PRAGMA table_info(conversation_states)")
            }
            if "signal_streak" not in state_columns:
                db.execute(
                    """ALTER TABLE conversation_states
                       ADD COLUMN signal_streak INTEGER NOT NULL DEFAULT 0"""
                )
            if "customer_id" not in state_columns:
                db.execute(
                    "ALTER TABLE conversation_states ADD COLUMN customer_id TEXT"
                )
            db.execute(
                """CREATE TABLE IF NOT EXISTS customer_states (
                    customer_id TEXT PRIMARY KEY,
                    last_reply_at REAL
                )"""
            )
            db.execute(
                """CREATE TABLE IF NOT EXISTS human_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    sender TEXT NOT NULL,
                    message TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    delivered INTEGER NOT NULL DEFAULT 0
                )"""
            )
            db.execute(
                """CREATE TABLE IF NOT EXISTS escalated_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    message TEXT NOT NULL,
                    last_agent_message TEXT,
                    created_at REAL NOT NULL
                )"""
            )

    def get_or_create_state(
        self, conversation_id: str, customer_id: str
    ) -> ConversationState:
        with self._connect() as db:
            db.execute(
                """INSERT OR IGNORE INTO conversation_states
                   (conversation_id, customer_id, status)
                   VALUES (?, ?, 'active')""",
                (conversation_id, customer_id),
            )
            row = db.execute(
                """SELECT status, signal_streak, customer_id
                   FROM conversation_states WHERE conversation_id = ?""",
                (conversation_id,),
            ).fetchone()
            stored_customer_id = row[2]
            if stored_customer_id is None:
                db.execute(
                    """UPDATE conversation_states SET customer_id = ?
                       WHERE conversation_id = ?""",
                    (customer_id, conversation_id),
                )
            elif stored_customer_id != customer_id:
                raise ValueError(
                    f"conversation {conversation_id!r} belongs to another customer"
                )

            db.execute(
                """INSERT OR IGNORE INTO customer_states
                   (customer_id, last_reply_at) VALUES (?, NULL)""",
                (customer_id,),
            )
            columns = {
                item[1] for item in db.execute("PRAGMA table_info(conversation_states)")
            }
            if "last_reply_at" in columns:
                legacy = db.execute(
                    """SELECT last_reply_at FROM conversation_states
                       WHERE conversation_id = ?""",
                    (conversation_id,),
                ).fetchone()[0]
                if legacy is not None:
                    db.execute(
                        """UPDATE customer_states
                           SET last_reply_at = MAX(COALESCE(last_reply_at, ?), ?)
                           WHERE customer_id = ?""",
                        (legacy, legacy, customer_id),
                    )
        return {
            "status": row[0],
            "signal_streak": row[1],
        }

    def set_status(self, conversation_id: str, status: str) -> None:
        with self._connect() as db:
            db.execute(
                """INSERT INTO conversation_states (conversation_id, status)
                   VALUES (?, ?)
                   ON CONFLICT(conversation_id) DO UPDATE SET status = excluded.status""",
                (conversation_id, status),
            )

    def set_signal_streak(self, conversation_id: str, value: int) -> None:
        with self._connect() as db:
            db.execute(
                """UPDATE conversation_states SET signal_streak = ?
                   WHERE conversation_id = ?""",
                (value, conversation_id),
            )

    def claim_send_slot(self, customer_id: str, now: float) -> tuple[bool, float]:
        """Atomically claim the customer's sliding-window send slot."""
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                """INSERT OR IGNORE INTO customer_states
                   (customer_id, last_reply_at) VALUES (?, NULL)""",
                (customer_id,),
            )
            last_reply_at = db.execute(
                "SELECT last_reply_at FROM customer_states WHERE customer_id = ?",
                (customer_id,),
            ).fetchone()[0]
            if last_reply_at is not None and now - last_reply_at < 60:
                return False, last_reply_at + 60
            db.execute(
                """UPDATE customer_states SET last_reply_at = ?
                   WHERE customer_id = ?""",
                (now, customer_id),
            )
        return True, now

    def list_conversations(self, status: str) -> list[str]:
        with self._connect() as db:
            rows = db.execute(
                """SELECT conversation_id FROM conversation_states
                   WHERE status = ? ORDER BY conversation_id""",
                (status,),
            ).fetchall()
        return [row[0] for row in rows]

    def get_customer_id(self, conversation_id: str) -> str | None:
        with self._connect() as db:
            row = db.execute(
                """SELECT customer_id FROM conversation_states
                   WHERE conversation_id = ?""",
                (conversation_id,),
            ).fetchone()
        return row[0] if row else None

    def enqueue(
        self,
        conversation_id: str,
        message: str,
        last_agent_message: str | None,
        ready_at: float,
        fixed_reply: str | None = None,
    ) -> None:
        with self._connect() as db:
            db.execute(
                """INSERT INTO pending_messages
                   (conversation_id, message, last_agent_message, ready_at, fixed_reply)
                   VALUES (?, ?, ?, ?, ?)""",
                (conversation_id, message, last_agent_message, ready_at, fixed_reply),
            )

    def next_ready(self, conversation_id: str, now: float) -> QueuedMessage | None:
        with self._connect() as db:
            row = db.execute(
                """SELECT id, message, last_agent_message, fixed_reply
                   FROM pending_messages
                   WHERE conversation_id = ? AND ready_at <= ?
                   ORDER BY id LIMIT 1""",
                (conversation_id, now),
            ).fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "message": row[1],
            "last_agent_message": row[2],
            "fixed_reply": row[3],
        }

    def delete(self, message_id: int) -> None:
        with self._connect() as db:
            db.execute("DELETE FROM pending_messages WHERE id = ?", (message_id,))

    def reschedule(self, message_id: int, ready_at: float) -> None:
        with self._connect() as db:
            db.execute(
                "UPDATE pending_messages SET ready_at = ? WHERE id = ?",
                (ready_at, message_id),
            )

    def clear(self, conversation_id: str) -> None:
        with self._connect() as db:
            db.execute(
                "DELETE FROM pending_messages WHERE conversation_id = ?",
                (conversation_id,),
            )

    def has_pending(self, conversation_id: str) -> bool:
        with self._connect() as db:
            row = db.execute(
                "SELECT 1 FROM pending_messages WHERE conversation_id = ? LIMIT 1",
                (conversation_id,),
            ).fetchone()
        return row is not None

    def save_escalated(
        self,
        conversation_id: str,
        message: str,
        last_agent_message: str | None,
        created_at: float,
    ) -> None:
        with self._connect() as db:
            db.execute(
                """INSERT INTO escalated_messages
                   (conversation_id, message, last_agent_message, created_at)
                   VALUES (?, ?, ?, ?)""",
                (conversation_id, message, last_agent_message, created_at),
            )

    def list_escalated(self, conversation_id: str) -> list[EscalatedMessage]:
        with self._connect() as db:
            rows = db.execute(
                """SELECT id, message, last_agent_message, created_at
                   FROM escalated_messages
                   WHERE conversation_id = ? ORDER BY id""",
                (conversation_id,),
            ).fetchall()
        return [
            {
                "id": row[0],
                "message": row[1],
                "last_agent_message": row[2],
                "created_at": row[3],
            }
            for row in rows
        ]

    def send_human_message(
        self,
        conversation_id: str,
        sender: Literal["user", "admin"],
        message: str,
        created_at: float,
    ) -> None:
        with self._connect() as db:
            db.execute(
                """INSERT INTO human_messages
                   (conversation_id, sender, message, created_at)
                   VALUES (?, ?, ?, ?)""",
                (conversation_id, sender, message, created_at),
            )

    def pop_human_message(
        self,
        conversation_id: str,
        sender: Literal["user", "admin"],
    ) -> HumanMessage | None:
        with self._connect() as db:
            row = db.execute(
                """SELECT id, sender, message, created_at FROM human_messages
                   WHERE conversation_id = ? AND sender = ? AND delivered = 0
                   ORDER BY id LIMIT 1""",
                (conversation_id, sender),
            ).fetchone()
            if row:
                db.execute(
                    "UPDATE human_messages SET delivered = 1 WHERE id = ?", (row[0],)
                )
        if not row:
            return None
        return {
            "id": row[0],
            "sender": row[1],
            "message": row[2],
            "created_at": row[3],
        }

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)
