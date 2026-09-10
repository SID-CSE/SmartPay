import hashlib
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DB_PATH = Path(__file__).resolve().parent / "data" / "smartpay_community.db"


class DataStore:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_schema()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS contributions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    submitted_at TEXT NOT NULL,
                    company_name TEXT NOT NULL,
                    title TEXT NOT NULL,
                    experience_years REAL NOT NULL,
                    location TEXT NOT NULL,
                    skills TEXT,
                    remote_work TEXT,
                    actual_salary REAL NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    model_predicted_at_submit REAL,
                    deviation_ratio REAL,
                    submitter_hash TEXT,
                    notes TEXT
                );
                CREATE TABLE IF NOT EXISTS model_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trained_at TEXT NOT NULL,
                    n_training_rows INTEGER,
                    n_contributions_included INTEGER,
                    test_r2 REAL,
                    test_mae REAL,
                    artifact_dir TEXT,
                    is_current INTEGER DEFAULT 0,
                    notes TEXT
                );
                CREATE TABLE IF NOT EXISTS lookup_overrides (
                    key_type TEXT NOT NULL,
                    key_value TEXT NOT NULL,
                    log_salary_mean REAL NOT NULL,
                    sample_count INTEGER NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (key_type, key_value)
                );
            """)

    def add_contribution(self, company_name, title, experience_years, location, skills,
                         remote_work, actual_salary, model_predicted_at_submit=None,
                         submitter_hash=None, notes=None):
        deviation_ratio = None
        status = "pending"
        if model_predicted_at_submit and model_predicted_at_submit > 0:
            deviation_ratio = actual_salary / model_predicted_at_submit
            status = "verified" if 0.3 <= deviation_ratio <= 3.0 else "flagged_review"
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO contributions (
                    submitted_at, company_name, title, experience_years, location, skills,
                    remote_work, actual_salary, status, model_predicted_at_submit,
                    deviation_ratio, submitter_hash, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (datetime.now(timezone.utc).isoformat(), company_name.strip(), title.strip(),
                  float(experience_years), location.strip(), skills or "", remote_work or "",
                  float(actual_salary), status, model_predicted_at_submit, deviation_ratio,
                  submitter_hash, notes))
        return status

    def get_contributions(self, status=None):
        with self._conn() as conn:
            if status:
                rows = conn.execute("SELECT * FROM contributions WHERE status = ? ORDER BY submitted_at DESC", (status,)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM contributions ORDER BY submitted_at DESC").fetchall()
        return [dict(row) for row in rows]

    def count_usable_contributions(self):
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM contributions WHERE status = 'verified'").fetchone()[0]

    def update_contribution_status(self, contribution_id, new_status):
        with self._conn() as conn:
            conn.execute("UPDATE contributions SET status = ? WHERE id = ?", (new_status, contribution_id))

    def delete_contribution(self, contribution_id):
        with self._conn() as conn:
            conn.execute("DELETE FROM contributions WHERE id = ?", (contribution_id,))

    def record_model_version(self, n_training_rows, n_contributions_included, test_r2, test_mae,
                             artifact_dir, notes=None):
        with self._conn() as conn:
            conn.execute("UPDATE model_versions SET is_current = 0")
            conn.execute("""
                INSERT INTO model_versions (
                    trained_at, n_training_rows, n_contributions_included, test_r2,
                    test_mae, artifact_dir, is_current, notes
                ) VALUES (?, ?, ?, ?, ?, ?, 1, ?)
            """, (datetime.now(timezone.utc).isoformat(), n_training_rows,
                  n_contributions_included, test_r2, test_mae, str(artifact_dir), notes))

    def get_model_history(self):
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM model_versions ORDER BY trained_at DESC").fetchall()
        return [dict(row) for row in rows]

    def get_lookup_overrides(self, key_type):
        with self._conn() as conn:
            rows = conn.execute("SELECT key_value, log_salary_mean, sample_count FROM lookup_overrides WHERE key_type = ?", (key_type,)).fetchall()
        return {row["key_value"]: {"mean": row["log_salary_mean"], "count": row["sample_count"]} for row in rows}

    def refresh_lookup_overrides(self):
        verified = self.get_contributions(status="verified")
        aggregates = {}
        for contribution in verified:
            values = {
                "companyName": contribution["company_name"],
                "title_norm": contribution["title"],
                "primary_location": contribution["location"].split(",")[0],
            }
            for key_type, value in values.items():
                aggregates.setdefault((key_type, value.strip().lower()), []).append(np.log1p(contribution["actual_salary"]))
        with self._conn() as conn:
            conn.execute("DELETE FROM lookup_overrides")
            now = datetime.now(timezone.utc).isoformat()
            conn.executemany("""
                INSERT INTO lookup_overrides (key_type, key_value, log_salary_mean, sample_count, updated_at)
                VALUES (?, ?, ?, ?, ?)
            """, [(key_type, key_value, float(np.mean(values)), len(values), now)
                  for (key_type, key_value), values in aggregates.items()])


def hash_submitter(identifier: str) -> str:
    return hashlib.sha256(identifier.encode()).hexdigest()[:16]
