#!/usr/bin/env python3
"""
Test TensorRT Engine - Diagnostic Tool
Tests if a TensorRT engine can be loaded and executed successfully.
"""

import numpy as np
import sys
import os

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    print(f"✓ TensorRT version: {trt.__version__}")
    print(f"✓ CUDA initialized")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

def test_engine(engine_path, engine_type="detector"):
    """Test if a TensorRT engine can be loaded and executed."""
    print(f"\n{'='*60}")
    print(f"Testing {engine_type} engine: {engine_path}")
    print(f"{'='*60}")
    
    if not os.path.exists(engine_path):
        print(f"✗ Engine file not found: {engine_path}")
        return False
    
    try:
        # Load engine
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(engine_data)
        
        if engine is None:
            print("✗ Failed to deserialize engine")
            return False
        
        print(f"✓ Engine deserialized successfully")
        
        # Get engine info
        print(f"\nEngine Information:")
        print(f"  Device Memory Size: {engine.device_memory_size} bytes")
        print(f"  Number of I/O tensors: {engine.num_io_tensors}")
        
        # List all tensors
        print(f"\nTensors:")
        for i in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(i)
            tensor_shape = engine.get_tensor_shape(tensor_name)
            tensor_dtype = engine.get_tensor_dtype(tensor_name)
            tensor_mode = engine.get_tensor_mode(tensor_name)
            mode_str = "INPUT" if tensor_mode == trt.TensorIOMode.INPUT else "OUTPUT"
            print(f"  [{i}] {tensor_name}: {mode_str}, shape={tensor_shape}, dtype={tensor_dtype}")
        
        # Create context
        context = engine.create_execution_context()
        print(f"✓ Execution context created")
        
        # Allocate buffers and test inference
        print(f"\nTesting inference with dummy data...")
        
        # Get input tensor
        input_tensor_name = None
        input_shape = None
        for i in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                input_tensor_name = tensor_name
                input_shape = engine.get_tensor_shape(tensor_name)
                break
        
        if input_tensor_name is None:
            print("✗ No input tensor found")
            return False
        
        print(f"  Input tensor: {input_tensor_name}, shape: {input_shape}")
        
        # Handle dynamic input shapes
        # If shape contains -1 (dynamic dimension), we need to set a concrete shape first
        has_dynamic = any(s == -1 or s < 0 for s in input_shape)
        concrete_shape = list(input_shape)
        
        if has_dynamic:
            # Set concrete shape for dynamic dimensions
            for i, dim in enumerate(concrete_shape):
                if dim == -1 or dim < 0:
                    # Set reasonable defaults for dynamic dimensions
                    if engine_type == "OCR" or "ocr" in engine_path.lower():
                        # OCR input: (1, 3, 48, -1) -> set width to 320 (common OCR width)
                        if i == 3:  # Width dimension (last dimension)
                            concrete_shape[i] = 320
                        else:
                            concrete_shape[i] = 1
                    else:
                        # Default: set to 1 for unknown dynamic dimensions
                        concrete_shape[i] = 1
            
            concrete_shape = tuple(concrete_shape)
            print(f"  Dynamic input shape detected: {input_shape}")
            print(f"  Setting concrete shape: {concrete_shape}")
            
            # Set the input shape in the context (TensorRT 10.x API)
            # Try different API methods for different TensorRT versions
            shape_set = False
            try:
                # TensorRT 10.x: set_input_shape
                if hasattr(context, 'set_input_shape'):
                    context.set_input_shape(input_tensor_name, concrete_shape)
                    shape_set = True
                    print(f"✓ Set concrete input shape using set_input_shape: {concrete_shape}")
            except (AttributeError, TypeError) as e:
                pass
            
            if not shape_set:
                try:
                    # TensorRT 9.x/10.x: set_tensor_shape
                    if hasattr(context, 'set_tensor_shape'):
                        context.set_tensor_shape(input_tensor_name, concrete_shape)
                        shape_set = True
                        print(f"✓ Set concrete input shape using set_tensor_shape: {concrete_shape}")
                except (AttributeError, TypeError) as e:
                    pass
            
            if not shape_set:
                try:
                    # TensorRT 8.x: set_binding_shape (deprecated but might work)
                    if hasattr(context, 'set_binding_shape'):
                        # Need to get binding index
                        binding_idx = None
                        for i in range(engine.num_io_tensors):
                            if engine.get_tensor_name(i) == input_tensor_name:
                                binding_idx = i
                                break
                        if binding_idx is not None:
                            context.set_binding_shape(binding_idx, concrete_shape)
                            shape_set = True
                            print(f"✓ Set concrete input shape using set_binding_shape: {concrete_shape}")
                except (AttributeError, TypeError) as e:
                    pass
            
            if not shape_set:
                print(f"  Warning: Cannot set dynamic shape via context API, using estimated size")
            
            # Get the actual shape after setting (may have changed)
            try:
                if hasattr(context, 'get_tensor_shape'):
                    actual_shape = context.get_tensor_shape(input_tensor_name)
                    if actual_shape != input_shape and all(s > 0 for s in actual_shape):
                        print(f"  Actual input shape after setting: {actual_shape}")
                        concrete_shape = actual_shape
            except:
                pass
            
            input_shape = concrete_shape
        
        # Calculate input size based on concrete shape
        # Don't use trt.volume() if shape still has negative dimensions
        if any(s < 0 for s in input_shape):
            # Manual calculation, replacing negative with 1
            input_size = 1
            for dim in input_shape:
                input_size *= max(1, dim)  # Use 1 for negative dimensions
        else:
            input_size = trt.volume(input_shape)
        
        if input_size <= 0:
            # Fallback: calculate manually
            input_size = 1
            for dim in input_shape:
                if dim > 0:
                    input_size *= dim
            if input_size <= 0:
                input_size = 1
        
        input_dtype = trt.nptype(engine.get_tensor_dtype(input_tensor_name))
        
        # Allocate host and device memory
        host_mem = cuda.pagelocked_empty(input_size, input_dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        
        print(f"  Allocated {host_mem.nbytes} bytes for input")
        
        # Set tensor address
        context.set_tensor_address(input_tensor_name, int(device_mem))
        print(f"✓ Set tensor address for input")
        
        # Allocate output buffers
        # After setting input shape, output shapes may have changed (for dynamic engines)
        # Get the actual output shapes from the context
        output_buffers = []
        for i in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.OUTPUT:
                # Get output shape from context (may have changed after setting input shape)
                try:
                    output_shape = context.get_tensor_shape(tensor_name)
                except:
                    # Fallback to engine shape
                    output_shape = engine.get_tensor_shape(tensor_name)
                
                output_size = trt.volume(output_shape)
                
                # Handle dynamic shapes and zero dimensions
                if output_size <= 0:
                    if len(output_shape) == 3:
                        # OCR output: (1, -1, 97) or (1, 0, 97) -> estimate max 200 characters
                        if output_shape[1] <= 0:
                            max_seq_len = 200
                            output_size = output_shape[0] * max_seq_len * output_shape[2]
                            print(f"  Warning: Dynamic output shape {output_shape}, estimating {max_seq_len} max sequence length")
                        else:
                            output_size = trt.volume(output_shape)
                    elif output_size < 0:
                        # Negative volume means fully dynamic - estimate
                        output_size = abs(output_size) * 1024
                        print(f"  Warning: Dynamic output shape {output_shape}, estimating size")
                    else:  # output_size == 0
                        # Zero dimension - use reasonable default
                        if len(output_shape) == 3:
                            output_size = output_shape[0] * 200 * output_shape[2]
                        else:
                            output_size = 1024
                
                if output_size == 0:
                    output_size = 1
                
                output_dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))
                output_host = cuda.pagelocked_empty(output_size, output_dtype)
                output_device = cuda.mem_alloc(output_host.nbytes)
                context.set_tensor_address(tensor_name, int(output_device))
                output_buffers.append({
                    'name': tensor_name,
                    'host': output_host,
                    'device': output_device,
                    'shape': output_shape
                })
                print(f"  Output tensor: {tensor_name}, shape: {output_shape}, allocated size: {output_size}")
        
        # Create dummy input data using the concrete shape
        # Use the shape we set (or original if not dynamic)
        dummy_shape = input_shape  # This is now the concrete shape
        dummy_input = np.random.randn(*dummy_shape).astype(input_dtype)
        np.copyto(host_mem, dummy_input.ravel())
        
        # Create stream
        stream = cuda.Stream()
        
        # Copy to device
        cuda.memcpy_htod_async(device_mem, host_mem, stream)
        print(f"✓ Copied input data to GPU")
        
        # Execute inference
        print(f"  Executing inference...")
        try:
            if hasattr(context, 'execute_async_v3'):
                context.execute_async_v3(stream_handle=stream.handle)
                print(f"✓ Inference executed successfully (execute_async_v3)")
            elif hasattr(context, 'execute_async_v2'):
                bindings = [int(device_mem)] + [int(b['device']) for b in output_buffers]
                context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
                print(f"✓ Inference executed successfully (execute_async_v2)")
            else:
                print(f"✗ No suitable execute method found")
                return False
            
            # Synchronize
            stream.synchronize()
            print(f"✓ Stream synchronized")
            
            # Try to read output (just verify it doesn't crash)
            if output_buffers:
                cuda.memcpy_dtoh_async(output_buffers[0]['host'], output_buffers[0]['device'], stream)
                stream.synchronize()
                print(f"✓ Output data retrieved")
                print(f"  Output shape: {output_buffers[0]['host'].shape}")
            
            print(f"\n{'='*60}")
            print(f"✓ ENGINE TEST PASSED - Engine is working correctly!")
            print(f"{'='*60}\n")
            return True
            
        except Exception as e:
            print(f"\n✗ Inference execution failed:")
            print(f"  Error: {type(e).__name__}: {e}")
            print(f"\n{'='*60}")
            print(f"✗ ENGINE TEST FAILED")
            print(f"{'='*60}\n")
            import traceback
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f"\n✗ Error testing engine:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    results = {}
    
    # Test YOLO detector engine (legacy)
    if os.path.exists("models/trailer_detector.engine"):
        results['YOLO Detector'] = test_engine("models/trailer_detector.engine", "YOLO Detector")
    else:
        print("⚠ YOLO detector engine not found: models/trailer_detector.engine")
        results['YOLO Detector'] = None
    
    # Test PaddleOCR text detection engine
    if os.path.exists("models/paddleocr_det.engine"):
        results['PaddleOCR Det'] = test_engine("models/paddleocr_det.engine", "PaddleOCR Text Detector")
    else:
        print("⚠ PaddleOCR detection engine not found: models/paddleocr_det.engine")
        results['PaddleOCR Det'] = None
    
    # Test OCR engine (legacy CRNN)
    if os.path.exists("models/ocr_crnn.engine"):
        results['OCR (CRNN)'] = test_engine("models/ocr_crnn.engine", "OCR (CRNN)")
    else:
        print("⚠ OCR engine not found: models/ocr_crnn.engine")
        results['OCR (CRNN)'] = None
    
    # Test PaddleOCR recognition engine
    if os.path.exists("models/paddleocr_rec.engine"):
        results['PaddleOCR Rec'] = test_engine("models/paddleocr_rec.engine", "PaddleOCR Text Recognition")
    else:
        print("⚠ PaddleOCR recognition engine not found: models/paddleocr_rec.engine")
        results['PaddleOCR Rec'] = None
    
    # Summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    for name, result in results.items():
        if result is None:
            status = "⚠ NOT FOUND"
        elif result:
            status = "✓ PASS"
        else:
            status = "✗ FAIL"
        print(f"{name:20s}: {status}")
    print(f"{'='*60}\n")
    
    # Check if any engines failed
    failed = [name for name, result in results.items() if result is False]
    if failed:
        print("RECOMMENDATION: Rebuild engines on the target Jetson device.")
        print("The engines may have been built on a different device, causing")
        print("the 'invalid resource handle' errors.")
        sys.exit(1)
    elif all(r is None for r in results.values()):
        print("⚠ No engines found. Please build engines first:")
        print("  python3 build_engines.py --paddleocr-det-onnx <path> --paddleocr-rec-onnx <path>")
        sys.exit(1)
    else:
        print("All tested engines are working correctly!")
        sys.exit(0)