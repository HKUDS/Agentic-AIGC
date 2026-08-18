"""SQLite persistence layer.

The bot is single-process and every query touches an indexed row or two, so the
storage API is plain synchronous `sqlite3` guarded by a lock. That keeps the
call sites readable and the tests trivial; if the bot ever needs to scale past
one process, this module is the only place that has to change.
"""

import os
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from .models import Profile, Reminder, Subscription, User, WeightLog, WorkoutLog

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    telegram_id INTEGER NOT NULL UNIQUE,
    first_name TEXT NOT NULL DEFAULT '',
    username TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    state TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS profiles (
    user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    gender TEXT NOT NULL DEFAULT 'other',
    age INTEGER,
    height_cm INTEGER,
    weight_kg REAL,
    goal TEXT NOT NULL DEFAULT 'keep_fit',
    level TEXT NOT NULL DEFAULT 'beginner',
    equipment TEXT NOT NULL DEFAULT 'none',
    days_per_week INTEGER NOT NULL DEFAULT 3,
    timezone_offset_minutes INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS subscriptions (
    user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    plan TEXT NOT NULL DEFAULT 'free',
    expires_at REAL,
    source TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS reminders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    time_local TEXT NOT NULL,
    days TEXT NOT NULL,
    text TEXT NOT NULL DEFAULT '',
    enabled INTEGER NOT NULL DEFAULT 1,
    next_fire_at REAL NOT NULL DEFAULT 0,
    last_fired_at REAL
);
CREATE INDEX IF NOT EXISTS idx_reminders_due ON reminders(enabled, next_fire_at);
CREATE INDEX IF NOT EXISTS idx_reminders_user ON reminders(user_id);

CREATE TABLE IF NOT EXISTS workout_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at REAL NOT NULL,
    workout_name TEXT NOT NULL DEFAULT '',
    duration_minutes INTEGER NOT NULL DEFAULT 0,
    difficulty INTEGER NOT NULL DEFAULT 0,
    note TEXT NOT NULL DEFAULT '',
    completed INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_workout_logs_user ON workout_logs(user_id, created_at);

CREATE TABLE IF NOT EXISTS weight_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at REAL NOT NULL,
    weight_kg REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_weight_logs_user ON weight_logs(user_id, created_at);

CREATE TABLE IF NOT EXISTS ai_usage (
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    day TEXT NOT NULL,
    used INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (user_id, day)
);

CREATE TABLE IF NOT EXISTS chat_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at REAL NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chat_history_user ON chat_history(user_id, id);

CREATE TABLE IF NOT EXISTS payments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at REAL NOT NULL,
    amount INTEGER NOT NULL DEFAULT 0,
    currency TEXT NOT NULL DEFAULT '',
    charge_id TEXT NOT NULL DEFAULT '',
    payload TEXT NOT NULL DEFAULT ''
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_payments_charge
    ON payments(charge_id) WHERE charge_id <> '';

CREATE TABLE IF NOT EXISTS promo_redemptions (
    code TEXT NOT NULL,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at REAL NOT NULL,
    PRIMARY KEY (code, user_id)
);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def utc_day_key(timestamp: Optional[float] = None, offset_minutes: int = 0) -> str:
    """Return the YYYY-MM-DD key of a timestamp in the user's local timezone."""
    moment = datetime.fromtimestamp(
        timestamp if timestamp is not None else time.time(), tz=timezone.utc
    ) + timedelta(minutes=offset_minutes)
    return moment.strftime("%Y-%m-%d")


class Storage:
    """All database access for the bot."""

    def __init__(self, path: str):
        self.path = path
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------
    # Users, profiles, subscriptions
    # ------------------------------------------------------------------

    def get_or_create_user(
        self, telegram_id: int, first_name: str = "", username: str = ""
    ) -> User:
        """Fetch a user by Telegram id, creating the row set on first contact."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM users WHERE telegram_id = ?", (telegram_id,)
            ).fetchone()
            if row is None:
                now = time.time()
                cursor = self._conn.execute(
                    "INSERT INTO users (telegram_id, first_name, username, created_at, state)"
                    " VALUES (?, ?, ?, ?, '')",
                    (telegram_id, first_name, username, now),
                )
                user_id = int(cursor.lastrowid)
                self._conn.execute(
                    "INSERT INTO profiles (user_id) VALUES (?)", (user_id,)
                )
                self._conn.execute(
                    "INSERT INTO subscriptions (user_id, plan) VALUES (?, 'free')",
                    (user_id,),
                )
                self._conn.commit()
                row = self._conn.execute(
                    "SELECT * FROM users WHERE id = ?", (user_id,)
                ).fetchone()
            elif first_name or username:
                # Names change; keep them fresh so admin exports stay readable.
                self._conn.execute(
                    "UPDATE users SET first_name = ?, username = ? WHERE id = ?",
                    (first_name or row["first_name"], username or row["username"], row["id"]),
                )
                self._conn.commit()
                row = self._conn.execute(
                    "SELECT * FROM users WHERE id = ?", (row["id"],)
                ).fetchone()

            return self._hydrate_user(row)

    def get_user_by_telegram_id(self, telegram_id: int) -> Optional[User]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM users WHERE telegram_id = ?", (telegram_id,)
            ).fetchone()
            return self._hydrate_user(row) if row else None

    def get_user(self, user_id: int) -> Optional[User]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM users WHERE id = ?", (user_id,)
            ).fetchone()
            return self._hydrate_user(row) if row else None

    def _hydrate_user(self, row: sqlite3.Row) -> User:
        """Assemble a `User` with its profile and subscription attached."""
        user_id = int(row["id"])
        profile_row = self._conn.execute(
            "SELECT * FROM profiles WHERE user_id = ?", (user_id,)
        ).fetchone()
        subscription_row = self._conn.execute(
            "SELECT * FROM subscriptions WHERE user_id = ?", (user_id,)
        ).fetchone()

        profile = Profile(user_id=user_id)
        if profile_row is not None:
            profile = Profile(
                user_id=user_id,
                gender=profile_row["gender"],
                age=profile_row["age"],
                height_cm=profile_row["height_cm"],
                weight_kg=profile_row["weight_kg"],
                goal=profile_row["goal"],
                level=profile_row["level"],
                equipment=profile_row["equipment"],
                days_per_week=profile_row["days_per_week"],
                timezone_offset_minutes=profile_row["timezone_offset_minutes"],
            )

        subscription = Subscription(user_id=user_id)
        if subscription_row is not None:
            subscription = Subscription(
                user_id=user_id,
                plan=subscription_row["plan"],
                expires_at=subscription_row["expires_at"],
                source=subscription_row["source"],
            )

        return User(
            id=user_id,
            telegram_id=int(row["telegram_id"]),
            first_name=row["first_name"],
            username=row["username"],
            created_at=row["created_at"],
            state=row["state"],
            profile=profile,
            subscription=subscription,
        )

    def set_state(self, user_id: int, state: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE users SET state = ? WHERE id = ?", (state, user_id)
            )
            self._conn.commit()

    def update_profile(self, user_id: int, **fields) -> None:
        """Update named profile columns. Unknown keys are ignored."""
        allowed = {
            "gender",
            "age",
            "height_cm",
            "weight_kg",
            "goal",
            "level",
            "equipment",
            "days_per_week",
            "timezone_offset_minutes",
        }
        updates = {key: value for key, value in fields.items() if key in allowed}
        if not updates:
            return
        assignments = ", ".join(f"{key} = ?" for key in updates)
        with self._lock:
            self._conn.execute(
                f"UPDATE profiles SET {assignments} WHERE user_id = ?",
                (*updates.values(), user_id),
            )
            self._conn.commit()

    def set_subscription(
        self,
        user_id: int,
        plan: str,
        expires_at: Optional[float],
        source: str = "",
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO subscriptions (user_id, plan, expires_at, source)"
                " VALUES (?, ?, ?, ?)"
                " ON CONFLICT(user_id) DO UPDATE SET"
                " plan = excluded.plan,"
                " expires_at = excluded.expires_at,"
                " source = excluded.source",
                (user_id, plan, expires_at, source),
            )
            self._conn.commit()

    def get_subscription(self, user_id: int) -> Optional[Subscription]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM subscriptions WHERE user_id = ?", (user_id,)
            ).fetchone()
        if row is None:
            return None
        return Subscription(
            user_id=user_id,
            plan=row["plan"],
            expires_at=row["expires_at"],
            source=row["source"],
        )

    # ------------------------------------------------------------------
    # AI usage quota
    # ------------------------------------------------------------------

    def ai_messages_used_today(self, user_id: int, offset_minutes: int = 0) -> int:
        day = utc_day_key(offset_minutes=offset_minutes)
        with self._lock:
            row = self._conn.execute(
                "SELECT used FROM ai_usage WHERE user_id = ? AND day = ?",
                (user_id, day),
            ).fetchone()
        return int(row["used"]) if row else 0

    def record_ai_message(self, user_id: int, offset_minutes: int = 0) -> int:
        """Increment today's AI counter and return the new value."""
        day = utc_day_key(offset_minutes=offset_minutes)
        with self._lock:
            self._conn.execute(
                "INSERT INTO ai_usage (user_id, day, used) VALUES (?, ?, 1)"
                " ON CONFLICT(user_id, day) DO UPDATE SET used = used + 1",
                (user_id, day),
            )
            self._conn.commit()
            row = self._conn.execute(
                "SELECT used FROM ai_usage WHERE user_id = ? AND day = ?",
                (user_id, day),
            ).fetchone()
        return int(row["used"])

    # ------------------------------------------------------------------
    # Chat history (context for the AI coach)
    # ------------------------------------------------------------------

    def append_chat_message(self, user_id: int, role: str, content: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO chat_history (user_id, created_at, role, content)"
                " VALUES (?, ?, ?, ?)",
                (user_id, time.time(), role, content),
            )
            self._conn.commit()

    def recent_chat(self, user_id: int, limit: int = 10) -> List[Dict[str, str]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content FROM chat_history WHERE user_id = ?"
                " ORDER BY id DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]

    def clear_chat(self, user_id: int) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM chat_history WHERE user_id = ?", (user_id,))
            self._conn.commit()

    # ------------------------------------------------------------------
    # Reminders
    # ------------------------------------------------------------------

    def add_reminder(
        self,
        user_id: int,
        kind: str,
        time_local: str,
        days: str,
        text: str,
        next_fire_at: float,
    ) -> Reminder:
        with self._lock:
            cursor = self._conn.execute(
                "INSERT INTO reminders (user_id, kind, time_local, days, text,"
                " enabled, next_fire_at) VALUES (?, ?, ?, ?, ?, 1, ?)",
                (user_id, kind, time_local, days, text, next_fire_at),
            )
            self._conn.commit()
            row = self._conn.execute(
                "SELECT * FROM reminders WHERE id = ?", (cursor.lastrowid,)
            ).fetchone()
        return self._hydrate_reminder(row)

    def list_reminders(self, user_id: int, only_enabled: bool = False) -> List[Reminder]:
        query = "SELECT * FROM reminders WHERE user_id = ?"
        params: List = [user_id]
        if only_enabled:
            query += " AND enabled = 1"
        query += " ORDER BY time_local, id"
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [self._hydrate_reminder(row) for row in rows]

    def get_reminder(self, reminder_id: int) -> Optional[Reminder]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM reminders WHERE id = ?", (reminder_id,)
            ).fetchone()
        return self._hydrate_reminder(row) if row else None

    def delete_reminder(self, reminder_id: int, user_id: int) -> bool:
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM reminders WHERE id = ? AND user_id = ?",
                (reminder_id, user_id),
            )
            self._conn.commit()
        return cursor.rowcount > 0

    def update_reminder_schedule(
        self,
        reminder_id: int,
        user_id: int,
        time_local: str,
        days: str,
        next_fire_at: float,
    ) -> bool:
        """Change when an existing reminder fires."""
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE reminders SET time_local = ?, days = ?, next_fire_at = ?"
                " WHERE id = ? AND user_id = ?",
                (time_local, days, next_fire_at, reminder_id, user_id),
            )
            self._conn.commit()
        return cursor.rowcount > 0

    def set_reminder_enabled(self, reminder_id: int, user_id: int, enabled: bool) -> bool:
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE reminders SET enabled = ? WHERE id = ? AND user_id = ?",
                (1 if enabled else 0, reminder_id, user_id),
            )
            self._conn.commit()
        return cursor.rowcount > 0

    def due_reminders(self, now: Optional[float] = None, limit: int = 100) -> List[Reminder]:
        moment = now if now is not None else time.time()
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM reminders WHERE enabled = 1 AND next_fire_at <= ?"
                " ORDER BY next_fire_at LIMIT ?",
                (moment, limit),
            ).fetchall()
        return [self._hydrate_reminder(row) for row in rows]

    def mark_reminder_fired(
        self, reminder_id: int, fired_at: float, next_fire_at: float
    ) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE reminders SET last_fired_at = ?, next_fire_at = ? WHERE id = ?",
                (fired_at, next_fire_at, reminder_id),
            )
            self._conn.commit()

    def reschedule_user_reminders(self, user_id: int, compute_next) -> None:
        """
        Recompute `next_fire_at` for every reminder of a user.

        Used after a timezone change, where every stored fire time shifts.
        `compute_next` takes a `Reminder` and returns the new timestamp.
        """
        for reminder in self.list_reminders(user_id):
            with self._lock:
                self._conn.execute(
                    "UPDATE reminders SET next_fire_at = ? WHERE id = ?",
                    (compute_next(reminder), reminder.id),
                )
                self._conn.commit()

    @staticmethod
    def _hydrate_reminder(row: sqlite3.Row) -> Reminder:
        return Reminder(
            id=int(row["id"]),
            user_id=int(row["user_id"]),
            kind=row["kind"],
            time_local=row["time_local"],
            days=row["days"],
            text=row["text"],
            enabled=bool(row["enabled"]),
            next_fire_at=row["next_fire_at"],
            last_fired_at=row["last_fired_at"],
        )

    # ------------------------------------------------------------------
    # Logs
    # ------------------------------------------------------------------

    def add_workout_log(
        self,
        user_id: int,
        workout_name: str,
        duration_minutes: int = 0,
        difficulty: int = 0,
        note: str = "",
        completed: bool = True,
        created_at: Optional[float] = None,
    ) -> WorkoutLog:
        moment = created_at if created_at is not None else time.time()
        with self._lock:
            cursor = self._conn.execute(
                "INSERT INTO workout_logs (user_id, created_at, workout_name,"
                " duration_minutes, difficulty, note, completed)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    user_id,
                    moment,
                    workout_name,
                    duration_minutes,
                    difficulty,
                    note,
                    1 if completed else 0,
                ),
            )
            self._conn.commit()
        return WorkoutLog(
            id=int(cursor.lastrowid),
            user_id=user_id,
            created_at=moment,
            workout_name=workout_name,
            duration_minutes=duration_minutes,
            difficulty=difficulty,
            note=note,
            completed=completed,
        )

    def list_workout_logs(
        self, user_id: int, since: Optional[float] = None, limit: int = 200
    ) -> List[WorkoutLog]:
        query = "SELECT * FROM workout_logs WHERE user_id = ?"
        params: List = [user_id]
        if since is not None:
            query += " AND created_at >= ?"
            params.append(since)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [
            WorkoutLog(
                id=int(row["id"]),
                user_id=int(row["user_id"]),
                created_at=row["created_at"],
                workout_name=row["workout_name"],
                duration_minutes=int(row["duration_minutes"]),
                difficulty=int(row["difficulty"]),
                note=row["note"],
                completed=bool(row["completed"]),
            )
            for row in rows
        ]

    def add_weight_log(
        self, user_id: int, weight_kg: float, created_at: Optional[float] = None
    ) -> WeightLog:
        moment = created_at if created_at is not None else time.time()
        with self._lock:
            cursor = self._conn.execute(
                "INSERT INTO weight_logs (user_id, created_at, weight_kg) VALUES (?, ?, ?)",
                (user_id, moment, weight_kg),
            )
            self._conn.commit()
        return WeightLog(
            id=int(cursor.lastrowid),
            user_id=user_id,
            created_at=moment,
            weight_kg=weight_kg,
        )

    def list_weight_logs(self, user_id: int, limit: int = 200) -> List[WeightLog]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM weight_logs WHERE user_id = ?"
                " ORDER BY created_at DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [
            WeightLog(
                id=int(row["id"]),
                user_id=int(row["user_id"]),
                created_at=row["created_at"],
                weight_kg=row["weight_kg"],
            )
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Payments and promo codes
    # ------------------------------------------------------------------

    def record_payment(
        self,
        user_id: int,
        amount: int,
        currency: str,
        charge_id: str,
        payload: str,
    ) -> bool:
        """
        Store a successful payment.

        Returns False when the charge was already recorded, which is how the
        bot stays idempotent if Telegram redelivers the same update.
        """
        try:
            with self._lock:
                self._conn.execute(
                    "INSERT INTO payments (user_id, created_at, amount, currency,"
                    " charge_id, payload) VALUES (?, ?, ?, ?, ?, ?)",
                    (user_id, time.time(), amount, currency, charge_id, payload),
                )
                self._conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def redeem_promo(self, user_id: int, code: str) -> bool:
        """Claim a promo code for a user. False when already redeemed."""
        try:
            with self._lock:
                self._conn.execute(
                    "INSERT INTO promo_redemptions (code, user_id, created_at)"
                    " VALUES (?, ?, ?)",
                    (code.upper(), user_id, time.time()),
                )
                self._conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    # ------------------------------------------------------------------
    # Bot-wide metadata
    # ------------------------------------------------------------------

    def get_meta(self, key: str) -> Optional[str]:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM meta WHERE key = ?", (key,)
            ).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, str(value)),
            )
            self._conn.commit()

    def count_users(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM users").fetchone()
        return int(row["n"])

    def count_pro_users(self, now: Optional[float] = None) -> int:
        moment = now if now is not None else time.time()
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM subscriptions WHERE plan = 'pro'"
                " AND (expires_at IS NULL OR expires_at > ?)",
                (moment,),
            ).fetchone()
        return int(row["n"])
