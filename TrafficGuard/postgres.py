"""
database/postgres.py
=======================
Thin PostgreSQL writer for alert events. Connection string comes
exclusively from settings.DATABASE_URL (sourced from .env) — never
hard-coded here.
"""

import logging

import psycopg2
import psycopg2.extras

logger = logging.getLogger("nids.db")

INSERT_SQL = """
INSERT INTO alerts (
    ts, risk_score, severity, src_ip, dst_ip, src_port, dst_port,
    protocol, attack_label, ml_confidence, rule_signature, rule_severity,
    raw_event
) VALUES (
    to_timestamp(%(ts)s), %(risk_score)s, %(severity)s, %(src_ip)s, %(dst_ip)s,
    %(src_port)s, %(dst_port)s, %(protocol)s, %(attack_label)s,
    %(ml_confidence)s, %(rule_signature)s, %(rule_severity)s, %(raw_event)s
)
"""


class PostgresWriter:
    def __init__(self, database_url: str):
        if not database_url:
            raise ValueError("DATABASE_URL is not set. Configure it in .env.")
        self.database_url = database_url
        self._conn = None

    def connect(self):
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self.database_url)
            self._conn.autocommit = True
            logger.info("Connected to PostgreSQL")
        return self._conn

    def insert_alert(self, risk_event: dict):
        conn = self.connect()
        ctx = risk_event.get("context", {})
        ml = risk_event.get("ml_evidence") or {}
        rule = risk_event.get("rule_evidence") or {}

        params = {
            "ts": risk_event.get("timestamp"),
            "risk_score": risk_event.get("risk_score"),
            "severity": risk_event.get("severity"),
            "src_ip": ctx.get("src_ip"),
            "dst_ip": ctx.get("dst_ip"),
            "src_port": ctx.get("src_port"),
            "dst_port": ctx.get("dst_port"),
            "protocol": ctx.get("protocol"),
            "attack_label": ml.get("attack_label"),
            "ml_confidence": ml.get("confidence"),
            "rule_signature": rule.get("signature"),
            "rule_severity": rule.get("severity"),
            "raw_event": psycopg2.extras.Json(risk_event),
        }

        with conn.cursor() as cur:
            cur.execute(INSERT_SQL, params)

    def close(self):
        if self._conn is not None and not self._conn.closed:
            self._conn.close()