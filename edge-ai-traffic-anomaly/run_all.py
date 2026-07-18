"""
run_all.py — Script chạy toàn bộ evaluation pipeline ET-SSL.

Chạy lần lượt:
1. Data preprocessing (synthetic hoặc thật)
2. Benchmark FP32 vs INT8 vs ONNX
3. Optimization comparison
4. Detection metrics evaluation
5. Zero-day test
6. Baseline comparison (ET-SSL vs RF vs K-Means)
7. Drift simulation + incremental learning

Usage:
    python run_all.py                          # Dùng synthetic data (nhanh)
    python run_all.py --dataset unsw_nb15      # Sau khi đã preprocess dataset thật
    python run_all.py --quick                  # Chạy nhanh (ít sample hơn)
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def run_pipeline(dataset: str = "synthetic", quick: bool = False):
    n_runs   = 200 if quick else 1000
    n_flows  = 500 if quick else 2000
    n_epochs = 3   if quick else 5

    print("=" * 70)
    print("🚀 ET-SSL Full Evaluation Pipeline")
    print(f"   Dataset: {dataset} | Quick: {quick}")
    print("=" * 70)

    all_results = {"dataset": dataset, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    t_start = time.time()

    # ----------------------------------------------------------------
    # Step 1: Data preprocessing (nếu cần)
    # ----------------------------------------------------------------
    print("\n" + "─"*60)
    print("STEP 1: Data Preprocessing")
    print("─"*60)
    if dataset == "synthetic":
        print("  ℹ️  Synthetic data — không cần preprocess")
    else:
        data_dir = Path(f"data/processed/{dataset}")
        if (data_dir / "X_test.npy").exists():
            print(f"  ✅ Processed data found: {data_dir}")
        else:
            print(f"  ⚠️  No processed data found.")
            print(f"  Run: python data/preprocess.py --dataset {dataset} --data_path data/raw/{dataset}/")
            print("  Continuing with synthetic data as fallback...")
            dataset = "synthetic"

    # ----------------------------------------------------------------
    # Step 2: Benchmark
    # ----------------------------------------------------------------
    print("\n" + "─"*60)
    print("STEP 2: Performance Benchmark (FP32 vs INT8 vs ONNX)")
    print("─"*60)
    try:
        from optimization.benchmark import run_full_benchmark
        bm = run_full_benchmark(n_runs=n_runs)
        all_results["benchmark"] = {b: r.get("single_sample", {}) for b, r in bm.items()}
        print("  ✅ Benchmark complete")
    except Exception as e:
        print(f"  ❌ Benchmark failed: {e}")

    # ----------------------------------------------------------------
    # Step 3: Optimization comparison
    # ----------------------------------------------------------------
    print("\n" + "─"*60)
    print("STEP 3: Optimization Comparison (with accuracy)")
    print("─"*60)
    try:
        from optimization.compare_before_after import compare_all
        opt = compare_all(dataset_name=dataset, n_benchmark_runs=n_runs // 2, plot=False)
        all_results["optimization"] = opt.get("models", {})
        print("  ✅ Optimization comparison complete")
    except Exception as e:
        print(f"  ❌ Optimization comparison failed: {e}")

    # ----------------------------------------------------------------
    # Step 4: Detection metrics
    # ----------------------------------------------------------------
    print("\n" + "─"*60)
    print("STEP 4: Detection Performance Metrics")
    print("─"*60)
    try:
        from evaluation.metrics import evaluate
        metrics = evaluate(dataset_name=dataset, backend="onnx", plot=False)
        all_results["detection_metrics"] = metrics
        print("  ✅ Metrics evaluation complete")
    except Exception as e:
        print(f"  ❌ Metrics failed: {e}")

    # ----------------------------------------------------------------
    # Step 5: Zero-day test
    # ----------------------------------------------------------------
    print("\n" + "─"*60)
    print("STEP 5: Zero-Day Detection Test")
    print("─"*60)
    try:
        from evaluation.zero_day_test import run_zero_day_test
        zd = run_zero_day_test(dataset_name=dataset, backend="onnx", n_zero_day=100 if quick else 200)
        all_results["zero_day"] = zd
        print("  ✅ Zero-day test complete")
    except Exception as e:
        print(f"  ❌ Zero-day test failed: {e}")

    # ----------------------------------------------------------------
    # Step 6: Baseline comparison
    # ----------------------------------------------------------------
    print("\n" + "─"*60)
    print("STEP 6: Baseline Comparison (ET-SSL vs RF vs K-Means)")
    print("─"*60)
    try:
        from evaluation.baseline_comparison import run_comparison
        baseline = run_comparison(dataset_name=dataset)
        all_results["baseline_comparison"] = baseline
        print("  ✅ Baseline comparison complete")
    except Exception as e:
        print(f"  ❌ Baseline comparison failed: {e}")

    # ----------------------------------------------------------------
    # Step 7: Drift simulation
    # ----------------------------------------------------------------
    print("\n" + "─"*60)
    print("STEP 7: Drift Simulation + Incremental Learning")
    print("─"*60)
    try:
        from evaluation.drift_simulation import run_drift_simulation
        drift = run_drift_simulation(n_epochs=n_epochs, n_flows_per_epoch=n_flows)
        all_results["drift_simulation"] = drift
        print("  ✅ Drift simulation complete")
    except Exception as e:
        print(f"  ❌ Drift simulation failed: {e}")

    # ----------------------------------------------------------------
    # Step 8: Robustness / evasion (readme Bảng 12)
    # ----------------------------------------------------------------
    print("\n" + "─"*60)
    print("STEP 8: Robustness / Evasion Test")
    print("─"*60)
    try:
        from evaluation.robustness_test import run_robustness_test
        robust = run_robustness_test(dataset_name=dataset, backend="onnx", n_samples=300 if quick else 500)
        all_results["robustness"] = robust
        print("  ✅ Robustness test complete")
    except Exception as e:
        print(f"  ❌ Robustness test failed: {e}")

    # ----------------------------------------------------------------
    # Step 9: Privacy validation (readme §9)
    # ----------------------------------------------------------------
    print("\n" + "─"*60)
    print("STEP 9: Privacy Validation")
    print("─"*60)
    try:
        from evaluation.privacy_audit import run_privacy_audit
        privacy = run_privacy_audit(verbose=True)
        all_results["privacy_audit"] = privacy
        print("  ✅ Privacy audit complete" if privacy["passed"] else "  ⚠️ Privacy audit có cảnh báo")
    except Exception as e:
        print(f"  ❌ Privacy audit failed: {e}")

    # ----------------------------------------------------------------
    # Final summary
    # ----------------------------------------------------------------
    elapsed = time.time() - t_start
    all_results["total_elapsed_sec"] = round(elapsed, 1)

    out_path = Path("evaluation/results/full_report.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n" + "="*70)
    print("✅ FULL PIPELINE COMPLETE!")
    print(f"   Total time: {elapsed:.1f}s")
    print(f"   Report: {out_path}")
    print("="*70)

    # Quick summary table
    print("\n📊 SUMMARY:")
    if "detection_metrics" in all_results:
        m = all_results["detection_metrics"]
        print(f"  ET-SSL Detection: Acc={m.get('accuracy','?'):.4f} | F1={m.get('f1','?'):.4f} | AUC={m.get('auc','?'):.4f}")
    if "zero_day" in all_results:
        z = all_results["zero_day"]
        print(f"  Zero-day TPR: {z.get('zero_day_avg_tpr','?'):.4f} | FP rate: {z.get('baseline_fp_rate','?'):.4f}")
    if "optimization" in all_results:
        for backend in ["fp32", "int8", "onnx"]:
            r = all_results["optimization"].get(backend, {})
            if r and "error" not in r:
                print(f"  {backend.upper()}: {r.get('mean_latency_ms','?'):.3f}ms | F1={r.get('f1','?'):.4f}")

    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="synthetic",
                        choices=["synthetic", "unsw_nb15", "cic_darknet2020"])
    parser.add_argument("--quick", action="store_true", help="Quick mode (fewer samples)")
    args = parser.parse_args()
    run_pipeline(dataset=args.dataset, quick=args.quick)
