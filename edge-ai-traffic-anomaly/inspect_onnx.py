"""
inspect_onnx.py — Chạy script này để xem model ONNX nhận input/output gì.

Cách chạy:
    python inspect_onnx.py model/onnx_models
"""
import sys
from pathlib import Path

model_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "model/onnx_models")

print(f"== Nội dung thư mục {model_dir} ==")
for f in sorted(model_dir.iterdir()):
    print(" -", f.name, f"({f.stat().st_size} bytes)")

onnx_files = list(model_dir.glob("*.onnx"))
if not onnx_files:
    print("\n[!] Không tìm thấy file .onnx nào trong thư mục này.")
    sys.exit(0)

import onnxruntime as ort

for onnx_path in onnx_files:
    print(f"\n== Model: {onnx_path.name} ==")
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    print("Inputs:")
    for inp in sess.get_inputs():
        print(f"  - name={inp.name!r}  shape={inp.shape}  dtype={inp.type}")

    print("Outputs:")
    for out in sess.get_outputs():
        print(f"  - name={out.name!r}  shape={out.shape}  dtype={out.type}")

print("\n== File JSON/PKL/NPY khác trong thư mục ==")
for pattern in ("*.json", "*.pkl", "*.npy", "*.npz", "*.yaml", "*.yml"):
    for f in model_dir.glob(pattern):
        print(" -", f.name)
