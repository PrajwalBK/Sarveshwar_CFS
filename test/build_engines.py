#!/usr/bin/env python3
"""
Build TensorRT Engines from ONNX Models

This script converts ONNX models to TensorRT engines optimized for Jetson Orin NX.
"""

import argparse
import os
import sys
from pathlib import Path

try:
    import tensorrt as trt
    print(f"✓ TensorRT version: {trt.__version__}")
except ImportError:
    print("✗ TensorRT not found. Install: pip install nvidia-tensorrt")
    sys.exit(1)


def build_engine(onnx_path: str, engine_path: str, precision: str = "fp16", workspace_gb: int = 4, 
                 min_shapes: dict = None, opt_shapes: dict = None, max_shapes: dict = None):
    """
    Build a TensorRT engine from an ONNX model.
    
    Args:
        onnx_path: Path to input ONNX model
        engine_path: Path to output TensorRT engine
        precision: "fp16" or "fp32"
        workspace_gb: GPU workspace memory in GB
        min_shapes: Minimum input shapes (dict of tensor_name: shape tuple)
        opt_shapes: Optimal input shapes (dict of tensor_name: shape tuple)
        max_shapes: Maximum input shapes (dict of tensor_name: shape tuple)
    """
    print(f"\n{'='*60}")
    print(f"Building TensorRT Engine")
    print(f"{'='*60}")
    print(f"Input ONNX: {onnx_path}")
    print(f"Output Engine: {engine_path}")
    print(f"Precision: {precision.upper()}")
    print(f"Workspace: {workspace_gb} GB")
    
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    
    # Create logger
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    # Create builder and network
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parse ONNX model
    print(f"\nParsing ONNX model...")
    with open(onnx_path, "rb") as model:
        if not parser.parse(model.read()):
            print("✗ Failed to parse ONNX model")
            for error in range(parser.num_errors):
                print(f"  Error {error}: {parser.get_error(error)}")
            raise RuntimeError("ONNX parsing failed")
    
    print(f"✓ ONNX model parsed successfully")
    
    # Print network info
    print(f"\nNetwork Information:")
    print(f"  Inputs: {network.num_inputs}")
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        print(f"    [{i}] {input_tensor.name}: shape={input_tensor.shape}, dtype={input_tensor.dtype}")
    
    print(f"  Outputs: {network.num_outputs}")
    for i in range(network.num_outputs):
        output_tensor = network.get_output(i)
        print(f"    [{i}] {output_tensor.name}: shape={output_tensor.shape}, dtype={output_tensor.dtype}")
    
    # Configure builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)  # Convert GB to bytes
    
    # Disable CUDNN tactics to avoid pooling/convolution issues with dynamic shapes
    # This is a workaround for "Cask" errors in TensorRT 10.x
    try:
        tactic_sources = config.get_tactic_sources()
        # Remove CUDNN from tactic sources if available
        if hasattr(trt, 'TacticSource'):
            # TensorRT 10.x - disable CUDNN
            config.set_tactic_sources(tactic_sources & ~(1 << int(trt.TacticSource.CUDNN)))
            print(f"\n✓ CUDNN tactics disabled (workaround for Cask errors)")
    except Exception as e:
        print(f"\n⚠ Could not disable CUDNN tactics: {e}")
    
    # Set precision
    if precision.lower() == "fp16":
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print(f"\n✓ FP16 precision enabled")
        else:
            print(f"\n⚠ FP16 not supported on this platform, using FP32")
    elif precision.lower() == "fp32":
        print(f"\n✓ FP32 precision enabled")
    else:
        raise ValueError(f"Unknown precision: {precision}. Use 'fp16' or 'fp32'")
    
    # Set dynamic shapes if provided
    if min_shapes or opt_shapes or max_shapes:
        profile = builder.create_optimization_profile()
        
        for i in range(network.num_inputs):
            input_tensor = network.get_input(i)
            tensor_name = input_tensor.name
            
            # Get shapes
            if min_shapes and tensor_name in min_shapes:
                min_shape = min_shapes[tensor_name]
            else:
                # Use network shape as default
                shape = list(input_tensor.shape)
                shape[0] = 1  # Batch size = 1
                min_shape = tuple(shape)
            
            if opt_shapes and tensor_name in opt_shapes:
                opt_shape = opt_shapes[tensor_name]
            else:
                opt_shape = min_shape
            
            if max_shapes and tensor_name in max_shapes:
                max_shape = max_shapes[tensor_name]
            else:
                max_shape = opt_shape
            
            profile.set_shape(tensor_name, min_shape, opt_shape, max_shape)
            print(f"  Dynamic shape for {tensor_name}: min={min_shape}, opt={opt_shape}, max={max_shape}")
        
        config.add_optimization_profile(profile)
    
    # Build engine
    print(f"\nBuilding TensorRT engine (this may take several minutes)...")
    print(f"  This process optimizes the model for your GPU architecture.")
    print(f"  Please be patient...")
    
    try:
        # TensorRT 10.x uses build_serialized_network instead of build_engine
        if hasattr(builder, 'build_serialized_network'):
            # TensorRT 10.x API
            serialized_engine = builder.build_serialized_network(network, config)
            if serialized_engine is None:
                raise RuntimeError("Engine build failed")
            
            print(f"✓ Engine built successfully")
            
            # Save engine directly from serialized network
            print(f"\nSaving engine to {engine_path}...")
            os.makedirs(os.path.dirname(engine_path) or ".", exist_ok=True)
            with open(engine_path, "wb") as f:
                f.write(serialized_engine)
        elif hasattr(builder, 'build_engine'):
            # TensorRT 8.x/9.x API
            engine = builder.build_engine(network, config)
            if engine is None:
                raise RuntimeError("Engine build failed")
            
            print(f"✓ Engine built successfully")
            
            # Save engine
            print(f"\nSaving engine to {engine_path}...")
            os.makedirs(os.path.dirname(engine_path) or ".", exist_ok=True)
            with open(engine_path, "wb") as f:
                f.write(engine.serialize())
        else:
            raise RuntimeError("Unsupported TensorRT version - no build method found")
    except Exception as e:
        print(f"\n✗ Engine build failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Get engine info
    engine_size = os.path.getsize(engine_path)
    print(f"✓ Engine saved ({engine_size / (1024*1024):.2f} MB)")
    
    print(f"\n{'='*60}")
    print(f"✓ BUILD COMPLETE")
    print(f"{'='*60}\n")
    
    return engine_path


def build_detector_engine(onnx_path: str, engine_path: str, precision: str = "fp16"):
    """Build detector engine with standard YOLO configuration."""
    return build_engine(
        onnx_path=onnx_path,
        engine_path=engine_path,
        precision=precision,
        workspace_gb=4,
        min_shapes={"images": (1, 3, 640, 640)},
        opt_shapes={"images": (1, 3, 640, 640)},
        max_shapes={"images": (1, 3, 640, 640)}
    )


def build_ocr_engine(onnx_path: str, engine_path: str, precision: str = "fp16"):
    """Build OCR engine with dynamic width support."""
    return build_engine(
        onnx_path=onnx_path,
        engine_path=engine_path,
        precision=precision,
        workspace_gb=2,
        min_shapes={"x": (1, 1, 32, 1)},      # Changed: 1 channel, height 32
        opt_shapes={"x": (1, 1, 32, 320)},    # Changed: 1 channel, height 32
        max_shapes={"x": (1, 1, 32, 320)}    # Changed: 1 channel, height 32
    )


def build_paddleocr_det_engine(onnx_path: str, engine_path: str, precision: str = "fp16"):
    """Build PaddleOCR text detection engine."""
    return build_engine(
        onnx_path=onnx_path,
        engine_path=engine_path,
        precision=precision,
        workspace_gb=4,
        min_shapes={"x": (1, 3, 640, 640)},   # PaddleOCR det typically uses 640x640
        opt_shapes={"x": (1, 3, 640, 640)},
        max_shapes={"x": (1, 3, 640, 640)}
    )


def build_paddleocr_rec_engine(onnx_path: str, engine_path: str, precision: str = "fp16"):
    """Build PaddleOCR text recognition engine with dynamic width support."""
    return build_engine(
        onnx_path=onnx_path,
        engine_path=engine_path,
        precision=precision,
        workspace_gb=2,
        min_shapes={"x": (1, 3, 48, 1)},      # PaddleOCR rec: 3 channels, height 48, dynamic width
        opt_shapes={"x": (1, 3, 48, 320)},
        max_shapes={"x": (1, 3, 48, 320)}
    )


def main():
    parser = argparse.ArgumentParser(description="Build TensorRT engines from ONNX models")
    parser.add_argument("--detector-onnx", type=str, help="Path to YOLO detector ONNX model")
    parser.add_argument("--ocr-onnx", type=str, help="Path to OCR ONNX model (legacy CRNN)")
    parser.add_argument("--paddleocr-det-onnx", type=str, help="Path to PaddleOCR detection ONNX model")
    parser.add_argument("--paddleocr-rec-onnx", type=str, help="Path to PaddleOCR recognition ONNX model")
    parser.add_argument("--output-dir", type=str, default="models", help="Output directory for engines")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp16", "fp32"], 
                       help="Precision: fp16 (faster) or fp32 (more accurate)")
    parser.add_argument("--fp16", action="store_const", const="fp16", dest="precision",
                       help="Use FP16 precision (shortcut for --precision fp16)")
    parser.add_argument("--fp32", action="store_const", const="fp32", dest="precision",
                       help="Use FP32 precision (shortcut for --precision fp32)")
    parser.add_argument("--workspace", type=int, default=4, help="Workspace memory in GB")
    
    args = parser.parse_args()
    
    if not any([args.detector_onnx, args.ocr_onnx, args.paddleocr_det_onnx, args.paddleocr_rec_onnx]):
        parser.error("At least one ONNX model must be provided")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Build YOLO detector engine (legacy)
    if args.detector_onnx:
        detector_engine = os.path.join(args.output_dir, "trailer_detector.engine")
        try:
            build_detector_engine(args.detector_onnx, detector_engine, args.precision)
        except Exception as e:
            print(f"\n✗ Failed to build YOLO detector engine: {e}")
            sys.exit(1)
    
    # Build OCR engine (legacy CRNN)
    if args.ocr_onnx:
        ocr_engine = os.path.join(args.output_dir, "ocr_crnn.engine")
        try:
            build_ocr_engine(args.ocr_onnx, ocr_engine, args.precision)
        except Exception as e:
            print(f"\n✗ Failed to build OCR engine: {e}")
            sys.exit(1)
    
    # Build PaddleOCR text detection engine
    if args.paddleocr_det_onnx:
        paddleocr_det_engine = os.path.join(args.output_dir, "paddleocr_det.engine")
        try:
            build_paddleocr_det_engine(args.paddleocr_det_onnx, paddleocr_det_engine, args.precision)
        except Exception as e:
            print(f"\n✗ Failed to build PaddleOCR detection engine: {e}")
            sys.exit(1)
    
    # Build PaddleOCR text recognition engine
    if args.paddleocr_rec_onnx:
        paddleocr_rec_engine = os.path.join(args.output_dir, "paddleocr_rec.engine")
        try:
            build_paddleocr_rec_engine(args.paddleocr_rec_onnx, paddleocr_rec_engine, args.precision)
        except Exception as e:
            print(f"\n✗ Failed to build PaddleOCR recognition engine: {e}")
            sys.exit(1)
    
    print("\n✓ All engines built successfully!")
    print("\nNext steps:")
    print("  1. Test engines: python3 test_engine.py")
    print("  2. Run application: python3 -m app.main_trt_demo")


if __name__ == "__main__":
    main()



