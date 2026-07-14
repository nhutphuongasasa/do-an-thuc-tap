import psycopg2
import os

database_url = "postgresql://neondb_owner:npg_KBUNdnvg5W1s@ep-icy-dream-at0vl5nc-pooler.c-9.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"

schema_sql = """
CREATE TABLE IF NOT EXISTS alerts (
    id SERIAL PRIMARY KEY,
    ts TIMESTAMP WITH TIME ZONE NOT NULL,
    risk_score FLOAT NOT NULL,
    severity VARCHAR(20) NOT NULL,
    src_ip VARCHAR(45) NOT NULL,
    dst_ip VARCHAR(45) NOT NULL,
    src_port INT NOT NULL,
    dst_port INT NOT NULL,
    protocol VARCHAR(10) NOT NULL,
    attack_label VARCHAR(100),
    ml_confidence FLOAT,
    rule_signature VARCHAR(255),
    rule_severity INT,
    raw_event JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
"""

try:
    print("Connecting to Neon PostgreSQL...")
    conn = psycopg2.connect(database_url)
    conn.autocommit = True
    with conn.cursor() as cur:
        print("Creating table alerts if not exists...")
        cur.execute(schema_sql)
        print("Table alerts created or verified.")
    conn.close()
except Exception as e:
    print("Database connection failed:", e)
