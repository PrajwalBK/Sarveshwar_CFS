#!/usr/bin/env python3
"""
GPU and CUDA Diagnostics Utility for Trailer Vision Edge.
Helps check if PyTorch, PyCUDA, and TensorRT are properly configured for GPU execution.
"""

import sys
import os
import platform

def main():
    print("=" * 60)
    print("           TRAILER VISION EDGE - GPU DIAGNOSTICS")
    print("=" * 60)
    print(f"OS Platform: {platform.system()} {platform.release()}")
    print(f"Python Version: {sys.version}")
    print(f"Working Directory: {os.getcwd()}")
    print("-" * 60)

    # 1. Check PyTorch and CUDA
    print("[1] Checking PyTorch & CUDA Support:")
    torch_available = False
    cuda_available = False
    try:
        import torch
        torch_available = True
        print(f"  ✓ PyTorch is installed (version: {torch.__version__})")
        try:
            print(f"  PyTorch Location: {torch.__file__}")
        except Exception:
            pass
        
        cuda_available = torch.cuda.is_available()
        print(f"  CUDA Available: {cuda_available}")
        
        if cuda_available:
            device_count = torch.cuda.device_count()
            print(f"  CUDA Device Count: {device_count}")
            for i in range(device_count):
                print(f"    - Device {i}: {torch.cuda.get_device_name(i)}")
                try:
                    props = torch.cuda.get_device_properties(i)
                    print(f"      Memory: {props.total_memory / (1024**3):.2f} GB")
                    print(f"      Compute Capability: {props.major}.{props.minor}")
                except Exception as e:
                    print(f"      Could not retrieve device properties: {e}")
        else:
            print("  ✗ PyTorch does NOT see any CUDA GPU.")
            print("    This usually means you have a CPU-only PyTorch build installed.")
    except ImportError:
        print("  ✗ PyTorch is not installed in this environment.")
    except Exception as e:
        print(f"  ✗ Error importing/checking PyTorch: {e}")

    print("-" * 60)

    # 2. Check PyCUDA and TensorRT
    print("[2] Checking TensorRT & PyCUDA:")
    trt_available = False
    pycuda_available = False
    
    try:
        import tensorrt as trt
        trt_available = True
        print(f"  ✓ TensorRT is installed (version: {trt.__version__})")
    except ImportError:
        print("  - TensorRT library is not installed (expected on Jetson, optional on desktop).")
    except Exception as e:
        print(f"  ✗ Error importing TensorRT: {e}")

    try:
        import pycuda.driver as cuda
        pycuda_available = True
        print("  ✓ PyCUDA is installed")
        try:
            import pycuda.autoinit
            print(f"  ✓ CUDA Context initialized successfully via PyCUDA")
        except Exception as e:
            print(f"  ✗ PyCUDA autoinit failed: {e}")
    except ImportError:
        print("  - PyCUDA is not installed (expected on Jetson, optional on desktop).")
    except Exception as e:
        print(f"  ✗ Error importing PyCUDA: {e}")

    print("-" * 60)

    # 3. Check EasyOCR
    print("[3] Checking EasyOCR:")
    try:
        import easyocr
        print(f"  ✓ EasyOCR is installed (version: {getattr(easyocr, '__version__', 'unknown')})")
    except ImportError:
        print("  - EasyOCR is not installed.")

    # 4. Check Ultralytics YOLO
    print("[4] Checking Ultralytics YOLO:")
    try:
        from ultralytics import YOLO
        import ultralytics
        print(f"  ✓ Ultralytics YOLO is installed (version: {ultralytics.__version__})")
    except ImportError:
        print("  - Ultralytics is not installed.")

    print("-" * 60)

    # Summary and Actionable Recommendations
    print("[5] Diagnostic Summary & Actions:")
    
    if cuda_available:
        print("  ✓ PyTorch CUDA is ready!")
        print("  You can run the application with GPU acceleration.")
        print("  Make sure your .env has: DEVICE=cuda")
    else:
        print("  ▲ GPU is NOT available for deep learning (PyTorch/YOLO/EasyOCR).")
        print("  To fix this, please re-install PyTorch with CUDA support:")
        if platform.system() == 'Windows':
            print("\n  For Windows (CUDA 12.1/12.4):")
            print("  1. Activate virtual env: venv\\Scripts\\activate")
            print("  2. Run: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
            print("  (Or if your GPU needs CUDA 11.8: https://download.pytorch.org/whl/cu118)")
        elif "tegra" in platform.release().lower() or os.path.exists("/etc/nv_tegra_release"):
            print("\n  For NVIDIA Jetson:")
            print("  Jetson uses specific JetPack builds of PyTorch. DO NOT install from PyPI.")
            print("  Please download and install PyTorch from NVIDIA's official Jetson wheels:")
            print("  https://developer.download.nvidia.com/compute/redist/jp/")
        else:
            print("\n  For Linux:")
            print("  Run: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
            
    print("=" * 60)

if __name__ == "__main__":
    main()
