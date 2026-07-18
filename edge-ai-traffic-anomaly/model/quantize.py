"""
quantize.py — Post-training quantization cho ET-SSL encoder.

Kỹ thuật: Dynamic Quantization FP32 → INT8 dùng PyTorch.
So sánh kích thước, latency, và accuracy trước/sau quantization.

Tham chiếu: Giai đoạn 3 trong kế hoạch dự án.

Usage:
    python model/quantize.py --model_dir path/to/model
"""

import argparse
import sys
import time
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.encoder import load_encoder, get_model_info, Encoder


def dynamic_quantize(encoder: Encoder) -> nn.Module:
    """
    Dynamic Quantization: weights INT8, activations float32.
    Phù hợp cho inference CPU — không cần calibration data.

    Note: Dynamic quantization chỉ áp dụng cho nn.Linear.
    """
    quantized = torch.quantization.quantize_dynamic(
        encoder,
        qconfig_spec={nn.Linear},
        dtype=torch.qint8,
    )
    return quantized


def save_quantized_model(
    quantized_model: nn.Module,
    output_path: str | Path,
) -> Path:
    """Lưu quantized model state dict."""
    output_path = Path(output_path)
    # Lưu toàn bộ model (không chỉ state_dict) vì quantized model có cấu trúc khác
    torch.save(quantized_model.state_dict(), output_path)
    return output_path


def compare_size(original_path: Path, quantized_path: Path) -> dict:
    """So sánh kích thước file."""
    orig_kb = original_path.stat().st_size / 1024
    quant_kb = quantized_path.stat().st_size / 1024
    reduction = (1 - quant_kb / orig_kb) * 100

    return {
        "original_kb": round(orig_kb, 1),
        "quantized_kb": round(quant_kb, 1),
        "size_reduction_pct": round(reduction, 1),
    }


def benchmark_latency(model: nn.Module, input_dim: int, n_runs: int = 500) -> dict:
    """Đo latency inference."""
    model.eval()
    dummy = torch.randn(1, input_dim)

    # Warmup
    with torch.no_grad():
        for _ in range(20):
            model(dummy)

    # Measure
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(dummy)
            times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    return {
        "mean_ms": round(float(np.mean(times)), 3),
        "median_ms": round(float(np.median(times)), 3),
        "p95_ms": round(float(np.percentile(times, 95)), 3),
        "throughput_fps": round(float(1000 / np.mean(times)), 1),
    }


def compare_outputs(
    fp32_model: nn.Module,
    int8_model: nn.Module,
    input_dim: int,
    n_samples: int = 100,
) -> dict:
    """
    So sánh output embedding giữa FP32 và INT8.
    Đo mức độ sai lệch do quantization.
    """
    fp32_model.eval()
    int8_model.eval()

    rng = np.random.default_rng(42)
    X = rng.standard_normal((n_samples, input_dim)).astype(np.float32)
    tensor = torch.from_numpy(X)

    with torch.no_grad():
        z_fp32 = fp32_model(tensor).numpy()
        z_int8 = int8_model(tensor).numpy()

    abs_diff = np.abs(z_fp32 - z_int8)
    rel_diff = abs_diff / (np.abs(z_fp32) + 1e-8)

    return {
        "mean_abs_diff": round(float(np.mean(abs_diff)), 6),
        "max_abs_diff": round(float(np.max(abs_diff)), 6),
        "mean_rel_diff_pct": round(float(np.mean(rel_diff)) * 100, 3),
    }


def run_quantization(model_dir: str, n_runs: int = 500) -> dict:
    """
    Pipeline đầy đủ: load FP32 → quantize → so sánh → lưu.

    Returns:
        dict chứa toàn bộ kết quả so sánh
    """
    model_dir = Path(model_dir)
    fp32_path = model_dir / "encoder_fp32.pt"
    int8_out_path = model_dir / "encoder_int8_dynamic.pt"

    print("=" * 60)
    print("⚡ ET-SSL Quantization (FP32 → Dynamic INT8)")
    print("=" * 60)

    # Load FP32
    print("\n📦 Loading FP32 model...")
    fp32 = load_encoder(
        weights_path=fp32_path,
        config_path=model_dir / "config.json",
        device="cpu",
    )
    info = get_model_info(fp32)
    input_dim = info["input_dim"]
    print(f"  input_dim={input_dim} | params={info['total_params']:,} | {info['estimated_size_mb']} MB")

    # Quantize
    print("\n🔄 Applying dynamic quantization (Linear layers → INT8)...")
    int8 = dynamic_quantize(fp32)
    print("  ✅ Quantization done")

    # Lưu
    print(f"\n💾 Saving quantized model to: {int8_out_path}")
    save_quantized_model(int8, int8_out_path)

    # So sánh kích thước
    print("\n📊 Size comparison:")
    size_cmp = compare_size(fp32_path, int8_out_path)
    print(f"  FP32: {size_cmp['original_kb']:.1f} KB")
    print(f"  INT8: {size_cmp['quantized_kb']:.1f} KB")
    print(f"  Reduction: {size_cmp['size_reduction_pct']:.1f}%")

    # Benchmark latency
    print(f"\n⏱️  Benchmarking latency ({n_runs} runs)...")
    fp32_lat = benchmark_latency(fp32, input_dim, n_runs)
    int8_lat = benchmark_latency(int8, input_dim, n_runs)

    speedup = fp32_lat["mean_ms"] / int8_lat["mean_ms"]
    print(f"\n  FP32: {fp32_lat['mean_ms']:.3f} ms/sample | {fp32_lat['throughput_fps']:.1f} fps")
    print(f"  INT8: {int8_lat['mean_ms']:.3f} ms/sample | {int8_lat['throughput_fps']:.1f} fps")
    print(f"  Speedup: {speedup:.2f}x")

    # So sánh output
    print("\n🔬 Output comparison (embedding drift due to quantization):")
    out_cmp = compare_outputs(fp32, int8, input_dim)
    print(f"  Mean |Δz|: {out_cmp['mean_abs_diff']:.6f}")
    print(f"  Max  |Δz|: {out_cmp['max_abs_diff']:.6f}")
    print(f"  Mean relative diff: {out_cmp['mean_rel_diff_pct']:.3f}%")

    results = {
        "model_dir": str(model_dir),
        "size": size_cmp,
        "latency_fp32": fp32_lat,
        "latency_int8": int8_lat,
        "speedup": round(speedup, 2),
        "output_drift": out_cmp,
        "int8_model_path": str(int8_out_path),
    }

    print("\n" + "=" * 60)
    print("✅ Quantization complete!")
    print(f"   Size ↓ {size_cmp['size_reduction_pct']:.1f}% | Speedup {speedup:.2f}x | Drift {out_cmp['mean_rel_diff_pct']:.3f}%")
    print("=" * 60)

    return results


if __name__ == "__main__":
    from configs.paths import get_model_dir

    parser = argparse.ArgumentParser(description="Quantize ET-SSL encoder FP32 → INT8")
    parser.add_argument(
        "--model_dir",
        default=str(get_model_dir()),
        help="Model directory",
    )
    parser.add_argument("--n_runs", type=int, default=500, help="Benchmark runs")
    args = parser.parse_args()

    results = run_quantization(args.model_dir, args.n_runs)

    # Save kết quả
    import json
    out_json = Path(args.model_dir) / "quantization_report.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n📄 Report saved: {out_json}")
