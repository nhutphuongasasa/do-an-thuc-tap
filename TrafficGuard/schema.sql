-- database/schema.sql
-- Run once against your PostgreSQL database to set up the alerts table.

CREATE TABLE IF NOT EXISTS alerts (
    id              BIGSERIAL PRIMARY KEY,
    ts              TIMESTAMPTZ NOT NULL DEFAULT now(),
    risk_score      NUMERIC(5,2) NOT NULL,
    severity        VARCHAR(16) NOT NULL,
    src_ip          INET,
    dst_ip          INET,
    src_port        INTEGER,
    dst_port        INTEGER,
    protocol        VARCHAR(16),
    attack_label    VARCHAR(128),
    ml_confidence   NUMERIC(5,4),
    rule_signature  TEXT,
    rule_severity   INTEGER,
    raw_event       JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_alerts_ts ON alerts (ts);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts (severity);
CREATE INDEX IF NOT EXISTS idx_alerts_src_ip ON alerts (src_ip);