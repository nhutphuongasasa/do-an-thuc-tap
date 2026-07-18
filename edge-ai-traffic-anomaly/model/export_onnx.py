"""
export_onnx.py — Export encoder FP32 sang ONNX format.

Script này tái tạo ONNX file từ encoder_fp32.pt.
Dùng khi cần export lại sau khi fine-tune hoặc kiểm tra tính hợp lệ.

Usage:
    python model/export_onnx.py
    python model/export_onnx.py --model_dir path/to/model --output encoder_new.onnx
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch

# Thêm project root vào path
sys.path.insert(0, str(Path(__file__).parent.parent))
from model.encoder import load_encoder, get_model_info


def export_onnx(
    model_dir: str,
    output_filename: str = "encoder_v5.onnx",
    opset: int = 17,
    validate: bool = True,
) -> Path:
    """
    Export encoder FP32 → ONNX.

    Args:
        model_dir: Thư mục chứa encoder_fp32.pt và config.json
        output_filename: Tên file ONNX output (trong cùng model_dir)
        opset: ONNX opset version (17 là mặc định an toàn)
        validate: Kiểm tra ONNX sau khi export

    Returns:
        Path đến file ONNX đã tạo
    """
    model_dir = Path(model_dir)
    output_path = model_dir / output_filename

    print(f"📦 Loading FP32 encoder from: {model_dir}")
    encoder = load_encoder(
        weights_path=model_dir / "encoder_fp32.pt",
        config_path=model_dir / "config.json",
        device="cpu",
    )
    encoder.eval()

    info = get_model_info(encoder)
    print(f"  input_dim={info['input_dim']}, embed_dim={info['embed_dim']}")
    print(f"  Params: {info['total_params']:,} | Size: {info['estimated_size_mb']} MB")

    # Dummy input cho export
    dummy_input = torch.randn(1, info["input_dim"])

    print(f"\n🔄 Exporting to ONNX (opset={opset})...")
    torch.onnx.export(
        encoder,
        dummy_input,
        str(output_path),
        opset_version=opset,
        input_names=["flow_features"],
        output_names=["embedding"],
        dynamic_axes={
            "flow_features": {0: "batch_size"},
            "embedding": {0: "batch_size"},
        },
        do_constant_folding=True,
        export_params=True,
    )
    print(f"  ✅ Saved: {output_path}")

    # File size
    size_mb = output_path.stat().st_size / 1e6
    print(f"  Size: {size_mb:.2f} MB")

    if validate:
        _validate_onnx(output_path, encoder, info["input_dim"])

    return output_path


def _validate_onnx(onnx_path: Path, torch_model: torch.nn.Module, input_dim: int):
    """
    Validate ONNX: kiểm tra model hợp lệ + so sánh output với PyTorch.
    """
    print("\n🔍 Validating ONNX...")

    # 1. Structural check
    try:
        import onnx
        model_proto = onnx.load(str(onnx_path))
        onnx.checker.check_model(model_proto)
        print("  ✅ ONNX structural check: OK")
        print(f"  Opset: {model_proto.opset_import[0].version}")
        print(f"  Inputs: {[i.name for i in model_proto.graph.input]}")
        print(f"  Outputs: {[o.name for o in model_proto.graph.output]}")
    except ImportError:
        print("  ⚠️ onnx package not installed — skipping structural check")

    # 2. Numerical check với ONNX Runtime
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        in_name = sess.get_inputs()[0].name

        test_input = np.random.randn(4, input_dim).astype(np.float32)

        # ONNX output
        onnx_out = sess.run(None, {in_name: test_input})[0]

        # PyTorch output
        with torch.no_grad():
            pt_out = torch_model(torch.from_numpy(test_input)).numpy()

        max_diff = np.max(np.abs(onnx_out - pt_out))
        print(f"  ✅ ONNX vs PyTorch max diff: {max_diff:.6f}")
        if max_diff > 1e-3:
            print("  ⚠️  Difference > 1e-3 — kiểm tra lại!")
        else:
            print("  ✅ Numerical match: OK")

    except ImportError:
        print("  ⚠️ onnxruntime not installed — skipping numerical check")


if __name__ == "__main__":
    from configs.paths import get_model_dir

    parser = argparse.ArgumentParser(description="Export ET-SSL encoder to ONNX")
    parser.add_argument(
        "--model_dir",
        default=str(get_model_dir()),
        help="Directory containing encoder_fp32.pt and config.json",
    )
    parser.add_argument(
        "--output",
        default="encoder_v5.onnx",
        help="Output ONNX filename (saved in model_dir)",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--no-validate", action="store_true", help="Skip validation")

    args = parser.parse_args()

    out = export_onnx(
        model_dir=args.model_dir,
        output_filename=args.output,
        opset=args.opset,
        validate=not args.no_validate,
    )
    print(f"\n🎉 Export complete: {out}")
