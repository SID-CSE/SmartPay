"""
data_store.py -- Persistent storage layer for SmartPay India.

Uses SQLite by default (zero setup, works great for local/self-hosted deployment).

IMPORTANT DEPLOYMENT NOTE:
If you deploy this to Streamlit Community Cloud (or any platform with an
ephemeral filesystem), the SQLite file will be WIPED on every app restart/redeploy
and contributions will be lost. For that environment, swap this class's internals
for a real cloud database (Postgres via Supabase/Neon, or Google Sheets via gspread)
-- the rest of the app only calls the public methods below, so the swap is isolated
to this one file.
"""
import sqlite3
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from contextlib import contextmanager

DB_PATH = Path(__file__).resolve().parent / "data" / "smartpay_community.db"
DB_PATH.parent.mkdir(exist_ok=True)


class DataStore:
    """Thin wrapper around SQLite. Swap this class's internals to change backend."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
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
            conn.execute("""
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
                )
            """)
            conn.execute("""
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
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS lookup_overrides (
                    key_type TEXT NOT NULL,
                    key_value TEXT NOT NULL,
                    log_salary_mean REAL NOT NULL,
                    sample_count INTEGER NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (key_type, key_value)
                )
            """)

    # ------------------------------------------------------------------
    # Contributions
    # ------------------------------------------------------------------
    def add_contribution(self, company_name, title, experience_years, location, skills,
                          remote_work, actual_salary, model_predicted_at_submit=None,
                          submitter_hash=None, notes=None):
        """Store a user-submitted real salary data point. Flags large deviations for review."""
        deviation_ratio = None
        status = 'pending'
        if model_predicted_at_submit and model_predicted_at_submit > 0:
            deviation_ratio = actual_salary / model_predicted_at_submit
            # Basic sanity gate: wildly implausible submissions get flagged, not silently trusted
            if 0.3 <= deviation_ratio <= 3.0:
                status = 'verified'
            else:
                status = 'flagged_review'

        with self._conn() as conn:
            conn.execute("""
                INSERT INTO contributions
                (submitted_at, company_name, title, experience_years, location, skills,
                 remote_work, actual_salary, status, model_predicted_at_submit, deviation_ratio,
                 submitter_hash, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(timezone.utc).isoformat(), company_name.strip(), title.strip(),
                float(experience_years), location.strip(), skills or '', remote_work or '',
                float(actual_salary), status, model_predicted_at_submit, deviation_ratio,
                submitter_hash, notes,
            ))
        return status

    def get_contributions(self, status=None):
        with self._conn() as conn:
            if status:
                rows = conn.execute("SELECT * FROM contributions WHERE status=? ORDER BY submitted_at DESC", (status,)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM contributions ORDER BY submitted_at DESC").fetchall()
            return [dict(r) for r in rows]

    def count_usable_contributions(self):
        """'Usable' = verified (passed the sanity gate). Flagged rows need manual review first."""
        with self._conn() as conn:
            row = conn.execute("SELECT COUNT(*) as n FROM contributions WHERE status='verified'").fetchone()
            return row['n']

    def update_contribution_status(self, contribution_id, new_status):
        with self._conn() as conn:
            conn.execute("UPDATE contributions SET status=? WHERE id=?", (new_status, contribution_id))

    def delete_contribution(self, contribution_id):
        with self._conn() as conn:
            conn.execute("DELETE FROM contributions WHERE id=?", (contribution_id,))

    # ------------------------------------------------------------------
    # Model version history
    # ------------------------------------------------------------------
    def record_model_version(self, n_training_rows, n_contributions_included, test_r2, test_mae,
                              artifact_dir, notes=None):
        with self._conn() as conn:
            conn.execute("UPDATE model_versions SET is_current=0")
            conn.execute("""
                INSERT INTO model_versions
                (trained_at, n_training_rows, n_contributions_included, test_r2, test_mae, artifact_dir, is_current, notes)
                VALUES (?, ?, ?, ?, ?, ?, 1, ?)
            """, (datetime.now(timezone.utc).isoformat(), n_training_rows, n_contributions_included,
                  test_r2, test_mae, str(artifact_dir), notes))

    def get_model_history(self):
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM model_versions ORDER BY trained_at DESC").fetchall()
            return [dict(r) for r in rows]

    def get_current_model_version(self):
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM model_versions WHERE is_current=1 ORDER BY trained_at DESC LIMIT 1").fetchone()
            return dict(row) if row else None

    # ------------------------------------------------------------------
    # Lightweight incremental lookup blending (see retrain_pipeline.py for full retrains)
    # ------------------------------------------------------------------
    def upsert_lookup_override(self, key_type, key_value, log_salary_mean, sample_count):
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO lookup_overrides (key_type, key_value, log_salary_mean, sample_count, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(key_type, key_value) DO UPDATE SET
                    log_salary_mean=excluded.log_salary_mean,
                    sample_count=excluded.sample_count,
                    updated_at=excluded.updated_at
            """, (key_type, key_value, log_salary_mean, sample_count, datetime.now(timezone.utc).isoformat()))

    def get_lookup_overrides(self, key_type):
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM lookup_overrides WHERE key_type=?", (key_type,)).fetchall()
            return {r['key_value']: {'mean': r['log_salary_mean'], 'count': r['sample_count']} for r in rows}

    def refresh_lookup_overrides(self):
        """
        Recomputes company/title/location overrides from ALL verified contributions.
        Call this right after any contribution's status changes (new submission auto-verified,
        or an admin manually approves a flagged one) so blending takes effect immediately.
        """
        import numpy as np
        verified = self.get_contributions(status='verified')
        if not verified:
            return
        agg = {'companyName': {}, 'title_norm': {}, 'primary_location': {}}
        for c in verified:
            log_sal = np.log1p(c['actual_salary'])
            for key_type, raw_val in [('companyName', c['company_name']), ('title_norm', c['title']),
                                       ('primary_location', c['location'].split(',')[0])]:
                norm = raw_val.strip().lower()
                bucket = agg[key_type].setdefault(norm, [])
                bucket.append(log_sal)
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            for key_type, entries in agg.items():
                for key_value, values in entries.items():
                    conn.execute("""
                        INSERT INTO lookup_overrides (key_type, key_value, log_salary_mean, sample_count, updated_at)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(key_type, key_value) DO UPDATE SET
                            log_salary_mean=excluded.log_salary_mean,
                            sample_count=excluded.sample_count,
                            updated_at=excluded.updated_at
                    """, (key_type, key_value, float(np.mean(values)), len(values), now))


def hash_submitter(identifier: str) -> str:
    """One-way hash for lightweight duplicate/spam tracking without storing raw identifiers."""
    return hashlib.sha256(identifier.encode()).hexdigest()[:16]
