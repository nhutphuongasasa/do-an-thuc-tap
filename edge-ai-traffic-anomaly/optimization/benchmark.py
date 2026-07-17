"""
benchmark.py — Đo hiệu năng inference trên laptop.

Tái hiện Bảng 6, 11 của bài báo nhưng trên môi trường CPU laptop.
So sánh FP32 vs INT8 vs ONNX về: latency, throughput, RAM, CPU%.

Giả lập ràng buộc "edge":
- CPU-only inference (không GPU)
- Đo RAM/CPU% dùng psutil
- (Optional) Giới hạn thread để mô phỏng CPU yếu

Usage:
    python optimization/benchmark.py
    python optimization/benchmark.py --n_runs 2000 --threads 2
"""

import argparse
import sys
import time
import gc
import json
import os
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.inference import ETSSLInference

MODEL_DIR = str(
    Path(__file__).parent.parent.parent / "TrafficGuard/models/edge_ai-20260716T101644Z-1-001/edge_ai"
)


# =====================================================================
# Resource monitoring
# =====================================================================
class ResourceMonitor:
    """Monitor CPU% và RAM usage trong quá trình inference."""

    def __init__(self):
        try:
            import psutil
            self.psutil = psutil
            self.process = psutil.Process(os.getpid())
            self.available = True
        except ImportError:
            self.available = False
            print("  ⚠️  psutil not installed — skipping resource monitoring")

    def sample(self) -> dict:
        if not self.available:
            return {"cpu_pct": None, "ram_mb": None}
        return {
            "cpu_pct": self.process.cpu_percent(interval=0.1),
            "ram_mb": self.process.memory_info().rss / 1e6,
        }

    def start_monitor(self, interval_sec: float = 0.5) -> list:
        """Bắt đầu sampling resources theo interval. Trả về list để append vào."""
        return []

    def get_summary(self, samples: list) -> dict:
        if not samples or not self.available:
            return {}
        cpus = [s["cpu_pct"] for s in samples if s["cpu_pct"] is not None]
        rams = [s["ram_mb"] for s in samples if s["ram_mb"] is not None]
        return {
            "avg_cpu_pct": round(float(np.mean(cpus)), 1) if cpus else None,
            "peak_cpu_pct": round(float(np.max(cpus)), 1) if cpus else None,
            "avg_ram_mb": round(float(np.mean(rams)), 1) if rams else None,
            "peak_ram_mb": round(float(np.max(rams)), 1) if rams else None,
        }


# =====================================================================
# Benchmark single backend
# =====================================================================
def benchmark_backend(
    backend: str,
    n_runs: int = 1000,
    batch_sizes: list = None,
    input_dim: int = 20,
    monitor: ResourceMonitor = None,
) -> dict:
    """
    Benchmark đầy đủ cho 1 backend.

    Returns dict với latency stats, throughput, resource usage.
    """
    if batch_sizes is None:
        batch_sizes = [1, 8, 32, 64]

    print(f"\n{'─'*50}")
    print(f"📦 Backend: {backend.upper()}")
    print(f"{'─'*50}")

    try:
        engine = ETSSLInference(model_dir=MODEL_DIR, backend=backend)
    except Exception as e:
        print(f"  ❌ Failed to load: {e}")
        return {"backend": backend, "error": str(e)}

    rng = np.random.default_rng(42)
    result = {"backend": backend}

    # --- Single-sample latency ---
    print(f"  ⏱️  Single-sample latency ({n_runs} runs)...")
    test_x = rng.standard_normal(input_dim).astype(np.float32) * 0.5

    # Warmup
    for _ in range(20):
        engine.predict(test_x)

    # Measure với resource monitoring
    res_samples = []
    times = []
    for i in range(n_runs):
        if monitor and monitor.available and i % 50 == 0:
            res_samples.append(monitor.sample())

        t0 = time.perf_counter()
        engine.predict(test_x)
        times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    result["single_sample"] = {
        "mean_ms":      round(float(np.mean(times)), 3),
        "median_ms":    round(float(np.median(times)), 3),
        "std_ms":       round(float(np.std(times)), 3),
        "p95_ms":       round(float(np.percentile(times, 95)), 3),
        "p99_ms":       round(float(np.percentile(times, 99)), 3),
        "min_ms":       round(float(np.min(times)), 3),
        "throughput_fps": round(float(1000 / np.mean(times)), 1),
    }

    if monitor:
        result["resources"] = monitor.get_summary(res_samples)

    print(f"    mean={result['single_sample']['mean_ms']:.3f}ms | "
          f"p95={result['single_sample']['p95_ms']:.3f}ms | "
          f"tput={result['single_sample']['throughput_fps']:.1f} fps")

    # --- Batch throughput ---
    print(f"  📊 Batch throughput by batch size...")
    result["batch"] = {}
    for bs in batch_sizes:
        test_batch = rng.standard_normal((bs, input_dim)).astype(np.float32)
        # Warmup
        for _ in range(5):
            engine.predict_batch(test_batch)
        # Measure
        batch_times = []
        for _ in range(max(50, n_runs // 10)):
            t0 = time.perf_counter()
            engine.predict_batch(test_batch)
            batch_times.append((time.perf_counter() - t0) * 1000)
        avg_total = np.mean(batch_times)
        fps = bs * 1000 / avg_total
        result["batch"][str(bs)] = {
            "total_ms": round(float(avg_total), 3),
            "per_sample_ms": round(float(avg_total / bs), 3),
            "throughput_fps": round(float(fps), 1),
        }
        print(f"    batch={bs:4d}: {avg_total:.2f}ms total | {fps:.1f} fps")

    # --- Model file size ---
    model_dir = Path(MODEL_DIR)
    size_info = {}
    for fname in ["encoder_fp32.pt", "encoder_int8.pt", "encoder_v5.onnx"]:
        fpath = model_dir / fname
        if fpath.exists():
            size_kb = fpath.stat().st_size / 1024
            size_info[fname] = round(size_kb, 1)
    result["model_sizes_kb"] = size_info

    return result


# =====================================================================
# Compare all backends
# =====================================================================
def run_full_benchmark(
    n_runs: int = 1000,
    threads: int = None,
) -> dict:
    """
    Benchmark tất cả backends và in bảng so sánh.

    Args:
        n_runs: Số lần inference để đo
        threads: Giới hạn số thread (None = dùng tất cả)
    """
    # Giới hạn thread nếu muốn giả lập edge CPU
    if threads is not None:
        import torch
        torch.set_num_threads(threads)
        os.environ["OMP_NUM_THREADS"] = str(threads)
        os.environ["MKL_NUM_THREADS"] = str(threads)
        print(f"🔒 CPU threads limited to {threads} (edge simulation)")

    print("=" * 60)
    print("📊 ET-SSL Full Benchmark — Laptop CPU")
    print(f"   n_runs={n_runs} | threads={threads or 'all'}")
    print("=" * 60)

    monitor = ResourceMonitor()
    gc.collect()

    results = {}
    for backend in ["fp32", "int8", "onnx"]:
        results[backend] = benchmark_backend(
            backend=backend,
            n_runs=n_runs,
            monitor=monitor,
        )

    # --- Summary table ---
    print(f"\n{'='*70}")
    print(f"  SUMMARY TABLE (Single-sample inference)")
    print(f"{'='*70}")
    print(f"  {'Backend':<10} {'Mean(ms)':>9} {'P95(ms)':>9} {'Throughput':>12} {'Size(KB)':>10}")
    print(f"  {'─'*58}")

    size_map = {"fp32": "encoder_fp32.pt", "int8": "encoder_int8.pt", "onnx": "encoder_v5.onnx"}
    for backend, res in results.items():
        if "error" in res:
            print(f"  {backend:<10} ERROR: {res['error'][:40]}")
            continue
        s = res["single_sample"]
        sizes = res.get("model_sizes_kb", {})
        sz = sizes.get(size_map.get(backend, ""), "—")
        print(f"  {backend.upper():<10} {s['mean_ms']:>9.3f} {s['p95_ms']:>9.3f} {s['throughput_fps']:>10.1f}/s {sz:>10}")

    # Speedup vs FP32
    if "fp32" in results and "error" not in results["fp32"]:
        fp32_mean = results["fp32"]["single_sample"]["mean_ms"]
        print(f"\n  Speedup vs FP32 baseline:")
        for backend, res in results.items():
            if "error" not in res and backend != "fp32":
                speedup = fp32_mean / res["single_sample"]["mean_ms"]
                print(f"    {backend.upper()}: {speedup:.2f}x")

    print(f"{'='*70}")

    # Paper reference
    print("\n  📄 Paper reference (RTX 3090, server GPU):")
    print(f"     Latency: 15–25 ms | Throughput: ~1500–1900 flows/s")
    print(f"  ℹ️  Laptop CPU results không thể so sánh trực tiếp với GPU server.")
    print(f"      So sánh ý nghĩa hơn: FP32 vs INT8 vs ONNX (relative improvement).")

    # Resource usage
    if monitor.available:
        print(f"\n  💻 Resource usage during benchmark:")
        for backend, res in results.items():
            if "resources" in res and res["resources"]:
                r = res["resources"]
                print(f"    {backend.upper()}: CPU={r.get('avg_cpu_pct')}% "
                      f"| RAM={r.get('avg_ram_mb')} MB")

    return results


# =====================================================================
# CLI
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark ET-SSL inference on laptop CPU")
    parser.add_argument("--n_runs", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=None,
                        help="Limit CPU threads (edge simulation)")
    parser.add_argument("--output", default="optimization/results/benchmark.json")
    args = parser.parse_args()

    results = run_full_benchmark(n_runs=args.n_runs, threads=args.threads)

    # Save
    out_path = Path(__file__).parent / "results/benchmark.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n💾 Results saved: {out_path}")
