"""
suricata/rule_event.py
========================
Normalized representation of a Suricata `eve.json` alert event.
"""

from dataclasses import dataclass


# Suricata severity: 1 = highest priority ... 3 = lowest (informational)
SEVERITY_TO_SCORE = {
    1: 90,
    2: 60,
    3: 30,
}


@dataclass
class RuleEvent:
    signature: str
    severity: int
    category: str
    source_ip: str
    destination_ip: str
    source_port: int
    destination_port: int
    protocol: str
    timestamp: str

    @property
    def rule_score(self) -> int:
        return SEVERITY_TO_SCORE.get(self.severity, 50)

    def to_dict(self) -> dict:
        return {
            "signature": self.signature,
            "severity": self.severity,
            "category": self.category,
            "source_ip": self.source_ip,
            "destination_ip": self.destination_ip,
            "source_port": self.source_port,
            "destination_port": self.destination_port,
            "protocol": self.protocol,
            "timestamp": self.timestamp,
            "rule_score": self.rule_score,
        }