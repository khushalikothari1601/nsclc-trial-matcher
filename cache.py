"""SQLite cache for parsed TrialCriteria, keyed by NCT ID + a hash of the source
eligibility text (plus the schema version) so a trial is only re-parsed by the LLM
when its eligibility text actually changes or the extraction schema changes shape."""

import hashlib
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from schema import SCHEMA_VERSION, TrialCriteria

DB_PATH = Path(__file__).parent / "data" / "cache.sqlite3"


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@contextmanager
def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trial_criteria (
                nct_id TEXT PRIMARY KEY,
                text_hash TEXT NOT NULL,
                schema_version INTEGER NOT NULL,
                model TEXT NOT NULL,
                criteria_json TEXT NOT NULL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        yield conn
        conn.commit()
    finally:
        conn.close()


def get_cached_criteria(nct_id: str, text_hash: str) -> TrialCriteria | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT text_hash, schema_version, criteria_json FROM trial_criteria WHERE nct_id = ?",
            (nct_id,),
        ).fetchone()

    if row is None:
        return None

    stored_hash, stored_version, criteria_json = row
    if stored_hash != text_hash or stored_version != SCHEMA_VERSION:
        return None

    return TrialCriteria.model_validate_json(criteria_json)


def save_criteria(nct_id: str, text_hash: str, criteria: TrialCriteria, model: str) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO trial_criteria (nct_id, text_hash, schema_version, model, criteria_json, updated_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(nct_id) DO UPDATE SET
                text_hash = excluded.text_hash,
                schema_version = excluded.schema_version,
                model = excluded.model,
                criteria_json = excluded.criteria_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            (nct_id, text_hash, SCHEMA_VERSION, model, criteria.model_dump_json()),
        )


def list_all_criteria() -> list[TrialCriteria]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT criteria_json FROM trial_criteria WHERE schema_version = ?",
            (SCHEMA_VERSION,),
        ).fetchall()
    return [TrialCriteria.model_validate_json(r[0]) for r in rows]


def cache_stats() -> dict:
    with _connect() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM trial_criteria WHERE schema_version = ?", (SCHEMA_VERSION,)
        ).fetchone()[0]
        last_updated = conn.execute("SELECT MAX(updated_at) FROM trial_criteria").fetchone()[0]
    return {"trial_count": count, "last_updated": last_updated}
