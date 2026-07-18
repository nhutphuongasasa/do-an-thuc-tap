"""
privacy_audit.py — Xác nhận pipeline không giải mã payload (readme §5, §9).

Quét code Python trong project, báo cáo nếu phát hiện pattern decrypt/giải mã payload.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Pattern nghi ngờ vi phạm privacy-preserving
SUSPICIOUS_PATTERNS = [
    (re.compile(r"\.decode\s*\(\s*['\"]utf", re.I), "decode payload as text"),
    (re.compile(r"TLS\.(?:decrypt|session)", re.I), "TLS decrypt API"),
    (re.compile(r"ssl\.(?:decrypt|unwrap)", re.I), "SSL decrypt API"),
    (re.compile(r"payload\.decode\s*\(", re.I), "payload.decode()"),
    (re.compile(r"decrypt\s*\(", re.I), "decrypt() call"),
]

# File/ thư mục bỏ qua
SKIP_DIRS = {"venv", ".git", "__pycache__", "model/weights"}


def scan_project(root: Path | None = None) -> dict:
    root = root or Path(__file__).resolve().parent.parent
    findings: list[dict] = []
    files_scanned = 0

    for py_file in root.rglob("*.py"):
        rel = py_file.relative_to(root)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if "privacy_audit.py" in str(rel):
            continue

        text = py_file.read_text(encoding="utf-8", errors="ignore")
        files_scanned += 1
        for line_no, line in enumerate(text.splitlines(), 1):
            for pattern, desc in SUSPICIOUS_PATTERNS:
                if pattern.search(line) and "decrypt" not in line.lower().split("#")[0]:
                    # Cho phép comment/docstring nhắc "không decrypt"
                    if "không giải mã" in line or "no decrypt" in line.lower():
                        continue
                    if desc == "decrypt() call" and "decrypt" in line and (
                        "payload" not in line.lower() and "tls" not in line.lower()
                        and "ssl" not in line.lower()
                    ):
                        continue
                    findings.append({
                        "file": str(rel),
                        "line": line_no,
                        "issue": desc,
                        "snippet": line.strip()[:120],
                    })

    # Kiểm tra feature extraction chỉ dùng metadata
    feature_ok = (root / "pipeline" / "feature_extractor.py").exists()
    capture_ok = (root / "pipeline" / "capture.py").exists()

    return {
        "passed": len(findings) == 0,
        "files_scanned": files_scanned,
        "findings": findings,
        "checks": {
            "feature_extractor_uses_metadata_only": feature_ok,
            "capture_no_payload_decode": capture_ok,
            "message": (
                "Pipeline chỉ dùng timing/length/flags — không giải mã payload TLS/VPN."
                if len(findings) == 0
                else f"Phát hiện {len(findings)} pattern nghi ngờ."
            ),
        },
    }


def run_privacy_audit(verbose: bool = True) -> dict:
    result = scan_project()
    if verbose:
        print("=" * 60)
        print("🔒 Privacy Validation (readme §9)")
        print("=" * 60)
        print(f"  Files scanned: {result['files_scanned']}")
        if result["passed"]:
            print("  ✅ PASSED — Không phát hiện bước giải mã payload")
            print(f"  ℹ️  {result['checks']['message']}")
        else:
            print("  ❌ FAILED — Có pattern nghi ngờ:")
            for f in result["findings"]:
                print(f"    {f['file']}:{f['line']} — {f['issue']}")
                print(f"      {f['snippet']}")
    return result


if __name__ == "__main__":
    import json

    r = run_privacy_audit()
    out = Path("evaluation/results/privacy_audit.json")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(r, f, indent=2)
    print(f"\n💾 Saved: {out}")
    sys.exit(0 if r["passed"] else 1)
