"""
TensorRT YOLO Detector Wrapper

This module provides a wrapper for YOLO detection using TensorRT engines.
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple
import os
import threading

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    # Don't import autoinit at module level - it creates a context per thread
    # We'll handle context activation in the infer method
    _cuda_context_initialized = True
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    _cuda_context_initialized = False
    print("Warning: TensorRT or PyCUDA not available. Install: pip install nvidia-tensorrt pycuda")
except Exception as e:
    TRT_AVAILABLE = False
    _cuda_context_initialized = False
    print(f"Warning: CUDA initialization failed: {e}")


class TrtEngineYOLO:
    """
    TensorRT YOLO detector wrapper.
    
    Supports YOLOv5/YOLOv8-style detection models with TensorRT optimization.
    """
    
    def __init__(self, engine_path: str, input_size: Tuple[int, int] = (640, 640), conf_threshold: float = 0.20):
        """
        Initialize TensorRT YOLO detector.
        
        Args:
            engine_path: Path to TensorRT engine file (.engine)
            input_size: Input image size (width, height)
            conf_threshold: Confidence threshold for detections (lowered to 0.20 for better detection)
        """
        self.engine_path = engine_path
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")
        
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT or PyCUDA not available. Cannot initialize detector.")
        
        # Initialize TensorRT runtime
        self.trt = trt
        self.cuda = cuda
        
        # Create TensorRT logger
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        
        # Load engine
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        
        # Create runtime and deserialize engine
        runtime = trt.Runtime(TRT_LOGGER)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize engine: {engine_path}")
        
        # Ensure CUDA context is active before creating execution context
        # This is important for proper resource initialization
        primary_ctx = None
        try:
            device = self.cuda.Device(0)
            primary_ctx = device.retrieve_primary_context()
            if primary_ctx is not None:
                primary_ctx.push()
        except:
            primary_ctx = None
        
        # Create execution context (should be done with CUDA context active)
        self.context = self.engine.create_execution_context()
        
        # Pop context if we pushed it
        try:
            if primary_ctx is not None:
                primary_ctx.pop()
        except:
            pass
        
        # Thread lock for CUDA operations (CUDA contexts are not thread-safe)
        self._inference_lock = threading.Lock()
        
        # Allocate buffers (using tensor-based API for TensorRT 8.x/9.x)
        self.inputs, self.outputs, self.input_names, self.output_names, self.bindings, self.stream = self._allocate_buffers()
        
        # Get input/output shapes
        if len(self.input_names) > 0:
            input_shape_raw = self.engine.get_tensor_shape(self.input_names[0])
            # Convert to tuple properly using same logic as _allocate_buffers
            try:
                if isinstance(input_shape_raw, np.ndarray):
                    self.input_shape = tuple(int(x) for x in input_shape_raw.flatten())
                elif isinstance(input_shape_raw, (list, tuple)):
                    self.input_shape = tuple(int(x) for x in input_shape_raw)
                elif isinstance(input_shape_raw, (int, np.integer, np.int64, np.int32)):
                    self.input_shape = (int(input_shape_raw),)
                else:
                    try:
                        self.input_shape = tuple(int(x) for x in input_shape_raw)
                    except (TypeError, ValueError):
                        self.input_shape = (int(input_shape_raw),) if input_shape_raw else ()
            except Exception:
                self.input_shape = tuple(input_shape_raw) if input_shape_raw else ()
        else:
            raise RuntimeError("No input tensors found in engine")
        
        if len(self.output_names) > 0:
            output_shape_raw = self.engine.get_tensor_shape(self.output_names[0])
            # Convert to tuple properly using same logic as _allocate_buffers
            try:
                if isinstance(output_shape_raw, np.ndarray):
                    self.output_shape = tuple(int(x) for x in output_shape_raw.flatten())
                elif isinstance(output_shape_raw, (list, tuple)):
                    self.output_shape = tuple(int(x) for x in output_shape_raw)
                elif isinstance(output_shape_raw, (int, np.integer, np.int64, np.int32)):
                    self.output_shape = (int(output_shape_raw),)
                else:
                    try:
                        self.output_shape = tuple(int(x) for x in output_shape_raw)
                    except (TypeError, ValueError):
                        self.output_shape = (int(output_shape_raw),) if output_shape_raw else ()
            except Exception:
                self.output_shape = tuple(output_shape_raw) if output_shape_raw else ()
        else:
            raise RuntimeError("No output tensors found in engine")
        
        print(f"Loaded TensorRT engine: {engine_path}")
        print(f"  Input tensor: {self.input_names[0]}, shape: {self.input_shape}")
        print(f"  Output tensor: {self.output_names[0]}, shape: {self.output_shape}")
    
    def _allocate_buffers(self):
        """Allocate GPU buffers for input and output using tensor-based API."""
        inputs = []
        outputs = []
        input_names = []
        output_names = []
        bindings = []  # Keep bindings list for execute_async_v2 compatibility
        
        # Create CUDA stream in the current context
        # pycuda.autoinit should have already created a context
        # Ensure context is current (important for multi-threaded environments)
        try:
            # Try to get current context
            current_ctx = cuda.Context.get_current()
            if current_ctx is None:
                # No context - autoinit should have created one, but try to initialize
                import pycuda.autoinit
                current_ctx = cuda.Context.get_current()
        except:
            # If we can't get context, try to initialize
            import pycuda.autoinit
            current_ctx = cuda.Context.get_current()
        
        # Create stream in the current context
        stream = cuda.Stream()
        
        # Get all tensor names from the engine
        # TensorRT 8.x+ uses num_io_tensors
        try:
            num_tensors = self.engine.num_io_tensors
        except AttributeError:
            # Fallback for older TensorRT versions (shouldn't happen with 8.x+)
            raise RuntimeError("TensorRT version too old. Requires TensorRT 8.x or later.")
        
        for i in range(num_tensors):
            try:
                tensor_name = self.engine.get_tensor_name(i)
                tensor_shape_raw = self.engine.get_tensor_shape(tensor_name)
                
                # Convert tensor_shape to a tuple/list properly
                # TensorRT may return numpy arrays or scalars, ensure it's a sequence
                try:
                    # First, try to convert to numpy array to handle all cases
                    import numpy as np
                    if isinstance(tensor_shape_raw, np.ndarray):
                        tensor_shape = tuple(int(x) for x in tensor_shape_raw.flatten())
                    elif isinstance(tensor_shape_raw, (list, tuple)):
                        tensor_shape = tuple(int(x) for x in tensor_shape_raw)
                    elif isinstance(tensor_shape_raw, (int, np.integer, np.int64, np.int32)):
                        # Single dimension - convert to tuple
                        tensor_shape = (int(tensor_shape_raw),)
                    else:
                        # Try to convert - might be a sequence-like object
                        try:
                            tensor_shape = tuple(int(x) for x in tensor_shape_raw)
                        except (TypeError, ValueError):
                            # Last resort: wrap in tuple
                            tensor_shape = (int(tensor_shape_raw),) if tensor_shape_raw else ()
                except Exception as shape_err:
                    print(f"  Error converting shape for {tensor_name}: {shape_err}, raw: {tensor_shape_raw}, type: {type(tensor_shape_raw)}")
                    # Fallback: try direct conversion
                    try:
                        tensor_shape = tuple(tensor_shape_raw) if tensor_shape_raw else ()
                    except:
                        tensor_shape = (1,)  # Default fallback
                
                tensor_dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
                
                # Calculate size and determine buffer shape
                size = trt.volume(tensor_shape)
                buffer_shape = tensor_shape  # Default: use tensor shape
                
                if size < 0:  # Dynamic shape - use a reasonable default
                    # For dynamic shapes, estimate based on input_size
                    if len(tensor_shape) >= 2:
                        # Assume batch dimension is 1, estimate from other dimensions
                        estimated_dims = [s if s > 0 else (self.input_size[0] if hasattr(self, 'input_size') else 640) for s in tensor_shape]
                        buffer_shape = tuple(estimated_dims)
                        size = np.prod(estimated_dims)
                    else:
                        buffer_shape = (1024,)
                        size = 1024
                
                # Ensure minimum size
                if size <= 0:
                    size = 1
                    buffer_shape = (1,)
                
                # Convert size to Python int (not numpy.int64)
                size = int(size)
                
                # Allocate host and device buffers
                # cuda.pagelocked_empty expects a shape tuple, not a scalar
                host_mem = cuda.pagelocked_empty(buffer_shape, tensor_dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                
                # Ensure device memory is valid (check pointer)
                if device_mem is None or int(device_mem) == 0:
                    raise RuntimeError(f"Failed to allocate device memory for tensor {tensor_name}")
                
                # Don't set tensor addresses here - set them right before inference
                # This ensures they're set in the correct CUDA context state
                # Setting them during initialization might cause issues when used from different threads
                
                # Also add to bindings list for execute_async_v2 compatibility
                bindings.append(int(device_mem))
                
                # Check if input or output
                tensor_mode = self.engine.get_tensor_mode(tensor_name)
                tensor_info = {
                    'name': tensor_name,
                    'host': host_mem, 
                    'device': device_mem, 
                    'shape': tensor_shape,  # Original tensor shape from engine
                    'buffer_shape': buffer_shape,  # Actual buffer shape (may differ for dynamic shapes)
                    'dtype': tensor_dtype
                }
                
                if tensor_mode == trt.TensorIOMode.INPUT:
                    inputs.append(tensor_info)
                    input_names.append(tensor_name)
                else:  # OUTPUT
                    outputs.append(tensor_info)
                    output_names.append(tensor_name)
            except Exception as e:
                print(f"Warning: Failed to process tensor {i}: {e}")
                continue
        
        if len(inputs) == 0:
            raise RuntimeError("No input tensors found in engine")
        if len(outputs) == 0:
            raise RuntimeError("No output tensors found in engine")
        
        return inputs, outputs, input_names, output_names, bindings, stream
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for YOLO input.
        
        Args:
            image: BGR image (H, W, 3)
            
        Returns:
            Preprocessed image tensor (1, 3, H, W) normalized to [0, 1]
        """
        h, w = self.input_size[1], self.input_size[0]
        img_resized = cv2.resize(image, (w, h))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype(np.float32) / 255.0
        img_transposed = np.transpose(img_normalized, (2, 0, 1))  # HWC -> CHW
        img_batched = np.expand_dims(img_transposed, axis=0)  # Add batch dimension
        return img_batched
    
    def infer(self, preprocessed: np.ndarray) -> np.ndarray:
        """
        Run TensorRT inference.
        
        Args:
            preprocessed: Preprocessed image tensor (1, 3, H, W)
            
        Returns:
            Detection results: [N, 6] array where each row is [x1, y1, x2, y2, conf, cls]
        """
        # Use lock for thread safety (CUDA operations are not thread-safe)
        # Ensure CUDA context is available in this thread
        # For multi-threaded PyCUDA, we need to push the primary context onto the stack
        context_pushed = False
        try:
            current_ctx = self.cuda.Context.get_current()
            if current_ctx is None:
                # No context in this thread - try to get the primary context from main thread
                primary_ctx = None
                try:
                    import sys
                    # Try both possible module names (when run as module vs script)
                    main_module = None
                    for module_name in ['app.main_trt_demo', '__main__']:
                        if module_name in sys.modules:
                            candidate = sys.modules[module_name]
                            if hasattr(candidate, '_cuda_primary_context'):
                                main_module = candidate
                                break
                    
                    if main_module is not None:
                        primary_ctx = main_module._cuda_primary_context
                    else:
                        pass
                except Exception as ctx_err:
                    pass
                
                if primary_ctx is not None:
                    try:
                        # Push the primary context onto this thread's context stack
                        primary_ctx.push()
                        context_pushed = True
                        current_ctx = self.cuda.Context.get_current()
                        if current_ctx is not None:
                            pass
                        else:
                            pass
                    except Exception as push_err:
                        pass
                        primary_ctx = None
                
                # If getting context from main thread fails, try autoinit as fallback
                if current_ctx is None:
                    try:
                        import pycuda.autoinit
                        current_ctx = self.cuda.Context.get_current()
                        if current_ctx is not None:
                            pass
                    except Exception as autoinit_err:
                        pass
                
                if current_ctx is None:
                    raise RuntimeError("CUDA context not available. Ensure CUDA context is initialized in main thread.")
        except Exception as ctx_err:
            # Last resort: try autoinit
            try:
                current_ctx = self.cuda.Context.get_current()
                if current_ctx is None:
                    raise RuntimeError(f"Failed to initialize CUDA context: {ctx_err}")
            except Exception as e2:
                raise RuntimeError(f"Failed to activate CUDA context. Error: {ctx_err}, {e2}")
        
        try:
            # Synchronize CUDA context before inference to avoid conflicts with PyTorch/EasyOCR
            # This is critical when EasyOCR or other PyTorch-based libraries have used the GPU
            try:
                import torch
                if torch.cuda.is_available():
                    # Synchronize PyTorch CUDA operations before TensorRT inference
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
            except:
                pass  # Ignore if torch not available
            
            # Synchronize PyCUDA context
            try:
                current_ctx = self.cuda.Context.get_current()
                if current_ctx is not None:
                    current_ctx.synchronize()
            except:
                pass
            
            # Context is now active, proceed with inference
            with self._inference_lock:
                if len(self.inputs) == 0:
                    raise RuntimeError("No input tensors available")
                
                input_tensor = self.inputs[0]
                input_name = input_tensor['name']
                
                # Prepare input data - ensure it matches the expected shape
                input_data = preprocessed.astype(input_tensor['dtype'])
                
                # Use buffer_shape (actual allocated buffer shape) not tensor_shape
                expected_shape = input_tensor.get('buffer_shape', input_tensor['shape'])
                
                # Reshape to match the host buffer shape
                if input_data.shape != expected_shape:
                    # Reshape if needed
                    if input_data.size == np.prod(expected_shape):
                        input_data = input_data.reshape(expected_shape)
                    else:
                        # Size mismatch - pad or truncate
                        expected_size = np.prod(expected_shape)
                        flat_data = input_data.ravel()
                        if flat_data.size > expected_size:
                            flat_data = flat_data[:expected_size]
                        else:
                            flat_data = np.pad(flat_data, (0, expected_size - flat_data.size))
                        input_data = flat_data.reshape(expected_shape)
                
                # Copy to host buffer (host buffer has buffer_shape)
                np.copyto(input_tensor['host'], input_data)
                
                # For dynamic shapes, set the concrete input shape in the context before inference
                # This is critical for TensorRT engines with dynamic dimensions
                original_shape = input_tensor['shape']
                if any(s == -1 or s < 0 for s in original_shape):
                    # Has dynamic dimensions - set concrete shape
                    concrete_input_shape = expected_shape  # Use the buffer_shape we calculated
                    try:
                        # Try TensorRT 10.x API
                        if hasattr(self.context, 'set_input_shape'):
                            self.context.set_input_shape(self.input_names[0], concrete_input_shape)
                        elif hasattr(self.context, 'set_tensor_shape'):
                            self.context.set_tensor_shape(self.input_names[0], concrete_input_shape)
                    except Exception as shape_err:
                        print(f"  Warning: Could not set dynamic input shape: {shape_err}")
                        # Continue anyway - might still work
                
                # Validate device memory pointers before setting addresses
                input_device_ptr = int(input_tensor['device'])
                if input_device_ptr == 0:
                    raise RuntimeError("Invalid input device memory pointer (null)")
                
                output_device_ptr = None
                if len(self.output_names) > 0:
                    output_device_ptr = int(self.outputs[0]['device'])
                    if output_device_ptr == 0:
                        raise RuntimeError("Invalid output device memory pointer (null)")
                
                # IMPORTANT: Set tensor addresses AFTER ensuring context is active
                # This ensures TensorRT can properly access the CUDA resources
                try:
                    self.context.set_tensor_address(self.input_names[0], input_device_ptr)
                    if output_device_ptr is not None:
                        self.context.set_tensor_address(self.output_names[0], output_device_ptr)
                except Exception as e:
                    print(f"Warning: Failed to set tensor addresses: {e}")
                    import traceback
                    traceback.print_exc()
                    raise RuntimeError(f"Cannot set tensor addresses: {e}")
                
                # Reuse the stream created during initialization
                # Creating new streams for each inference can cause resource exhaustion and "Cask" errors
                # The stream is created in the same context during initialization, so it should be safe to reuse
                inference_stream = self.stream
                if inference_stream is None:
                    raise RuntimeError("CUDA stream not initialized")
                
                # Transfer input data to GPU
                try:
                    self.cuda.memcpy_htod_async(
                        input_tensor['device'],
                        input_tensor['host'],
                        inference_stream
                    )
                except Exception as copy_err:
                    print(f"Error copying data to GPU: {copy_err}")
                    raise RuntimeError(f"Failed to copy input data to GPU: {copy_err}")
                
                # Run inference - TensorRT 10.x uses execute_async_v3
                # When using set_tensor_address(), tensor addresses are already set
                # execute_async_v3 only needs the stream handle
                if hasattr(self.context, 'execute_async_v3'):
                    # TensorRT 10.x - use execute_async_v3 (tensor addresses already set via set_tensor_address)
                    # Ensure stream handle is valid
                    if inference_stream is None or inference_stream.handle is None:
                        raise RuntimeError("Invalid CUDA stream handle")
                    
                    # Execute inference
                    try:
                        # Ensure stream is synchronized before execution
                        inference_stream.synchronize()
                        
                        self.context.execute_async_v3(stream_handle=inference_stream.handle)
                    except RuntimeError as e:
                        error_str = str(e).lower()
                        # Check for "Cask" error or other CUDA/TensorRT errors
                        if "cask" in error_str or "invalid resource handle" in error_str or "cuda" in error_str or "cutensor" in error_str:
                            # CUDA context issue - try to recover by re-synchronizing
                            try:
                                import torch
                                import time
                                
                                # Synchronize PyTorch CUDA operations
                                if torch.cuda.is_available():
                                    torch.cuda.synchronize()
                                    torch.cuda.empty_cache()
                                
                                # Synchronize PyCUDA context
                                current_ctx = self.cuda.Context.get_current()
                                if current_ctx is not None:
                                    current_ctx.synchronize()
                                
                                # Small delay to let context stabilize
                                time.sleep(0.001)
                                
                                # Re-synchronize stream
                                inference_stream.synchronize()
                                
                                # Re-set tensor addresses (they might have been invalidated)
                                self.context.set_tensor_address(self.input_names[0], int(input_tensor['device']))
                                if len(self.output_names) > 0:
                                    self.context.set_tensor_address(self.output_names[0], int(self.outputs[0]['device']))
                                
                                # Retry inference
                                self.context.execute_async_v3(stream_handle=inference_stream.handle)
                            except Exception as e2:
                                raise RuntimeError(f"TensorRT inference failed after CUDA sync retry. Original error: {e}, Retry error: {e2}")
                        else:
                            raise
                elif hasattr(self.context, 'execute_async_v2'):
                    # TensorRT 8.x - use execute_async_v2 with bindings
                    self.context.execute_async_v2(bindings=self.bindings, stream_handle=inference_stream.handle)
                elif hasattr(self.context, 'execute_v2'):
                    # Synchronous version
                    self.context.execute_v2(bindings=self.bindings)
                    inference_stream.synchronize()
                else:
                    raise RuntimeError(f"No suitable execute method found. Available methods: {[m for m in dir(self.context) if 'execute' in m.lower()]}")
                
                # Transfer predictions back from GPU
                if len(self.outputs) == 0:
                    raise RuntimeError("No output tensors available")
                
                output_tensor = self.outputs[0]
                self.cuda.memcpy_dtoh_async(
                    output_tensor['host'],
                    output_tensor['device'],
                    inference_stream
                )
                
                # Synchronize
                inference_stream.synchronize()
                
                # Get output and reshape
                output_tensor = self.outputs[0]
                output = output_tensor['host']
                output_shape = output_tensor['shape']
                
                # Reshape output based on engine output shape
                # YOLOv8 TensorRT output: [batch, 84, 8400] where 84 = 4 bbox + 80 classes
                # YOLOv5 TensorRT output: [batch, 25200, 85] where 85 = 4 bbox + 1 conf + 80 classes
                
                if len(output_shape) == 3:
                    # [batch, features, num_detections] or [batch, num_detections, features]
                    output = output.reshape(output_shape)
                    output = output[0]  # Remove batch dimension -> [features, num_detections] or [num_detections, features]
                elif len(output_shape) == 2:
                    # [N, features] - already 2D
                    output = output.reshape(output_shape)
                
                # Detect YOLOv8 transposed format: [84, N] instead of [N, 84]
                # YOLOv8 has 84 features (4 bbox + 80 classes), YOLOv5 has 85 features (4 bbox + 1 conf + 80 classes)
                if output.shape[0] == 84 or (output.shape[0] < 100 and output.shape[1] > 1000):
                    # YOLOv8 transposed format: [84, 8400] -> transpose to [8400, 84]
                    output = output.T  # Now [num_detections, 84]
                
                # Handle different output formats
                if output.shape[1] == 84:
                    # YOLOv8 format: [N, 84] where 84 = 4 bbox coords + 80 class scores
                    # YOLOv8 doesn't have separate confidence, it's max(class_scores)
                    
                    boxes = output[:, :4]  # [x, y, w, h] in center format
                    class_scores = output[:, 4:]  # [80 classes]
                    
                    # Get confidence as max class score
                    max_scores = class_scores.max(axis=1)  # [N]
                    class_ids = class_scores.argmax(axis=1)  # [N]
                    
                    # YOLOv8 scores might not be normalized - check and normalize if needed
                    if max_scores.max() > 1.0:
                        # Scores are not in [0, 1] range, apply sigmoid to normalize
                        max_scores = 1 / (1 + np.exp(-max_scores))  # Sigmoid normalization
                    
                    # Convert center format (x_center, y_center, w, h) to corner format (x1, y1, x2, y2)
                    x_center, y_center, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
                    x1 = x_center - w / 2
                    y1 = y_center - h / 2
                    x2 = x_center + w / 2
                    y2 = y_center + h / 2
                    
                    # Combine into [x1, y1, x2, y2, conf, cls]
                    detections = np.column_stack([x1, y1, x2, y2, max_scores, class_ids])
                    
                elif output.shape[1] == 85:
                    # YOLOv5 format: [N, 85] where 85 = 4 bbox + 1 conf + 80 classes
                    
                    boxes = output[:, :4]  # [x, y, w, h]
                    obj_conf = output[:, 4]  # objectness confidence
                    class_scores = output[:, 5:]  # [80 classes]
                    
                    # Get class with highest score
                    class_ids = class_scores.argmax(axis=1)
                    class_confs = class_scores.max(axis=1)
                    
                    # Final confidence = obj_conf * class_conf
                    final_conf = obj_conf * class_confs
                    
                    # Convert center+size to corner format
                    x_center, y_center, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
                    x1 = x_center - w / 2
                    y1 = y_center - h / 2
                    x2 = x_center + w / 2
                    y2 = y_center + h / 2
                    
                    # Combine into [x1, y1, x2, y2, conf, cls]
                    detections = np.column_stack([x1, y1, x2, y2, final_conf, class_ids])
                    
                elif output.shape[1] == 6:
                    # Already in [x1, y1, x2, y2, conf, cls] format
                    detections = output
                elif output.shape[1] == 5:
                    # Single-class YOLOv8 format: [N, 5] where 5 = 4 bbox + 1 class score
                    # This is for models trained with nc=1 (single class detection)
                    
                    boxes = output[:, :4]  # [x, y, w, h] in center format
                    class_scores = output[:, 4]  # [1 class]
                    
                    # All detections are class 0 (the single class)
                    class_ids = np.zeros(len(output), dtype=np.float32)
                    
                    # YOLOv8 scores might not be normalized - check and normalize if needed
                    if class_scores.max() > 1.0:
                        # Scores are not in [0, 1] range, apply sigmoid to normalize
                        class_scores = 1 / (1 + np.exp(-class_scores))  # Sigmoid normalization
                    
                    # Convert center format (x_center, y_center, w, h) to corner format (x1, y1, x2, y2)
                    x_center, y_center, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
                    x1 = x_center - w / 2
                    y1 = y_center - h / 2
                    x2 = x_center + w / 2
                    y2 = y_center + h / 2
                    
                    # Combine into [x1, y1, x2, y2, conf, cls]
                    detections = np.column_stack([x1, y1, x2, y2, class_scores, class_ids])
                else:
                    # Unknown format - try to handle gracefully
                    if output.shape[1] >= 6:
                        detections = output[:, :6]
                    else:
                        # Pad with zeros for missing columns (class id)
                        padding = np.zeros((output.shape[0], 6 - output.shape[1]))
                        detections = np.hstack([output, padding])
                
                # Debug: Show confidence range before filtering
                if len(detections) > 0:
                    conf_min = detections[:, 4].min()
                    conf_max = detections[:, 4].max()
                    conf_mean = detections[:, 4].mean()
                
                # Apply confidence threshold
                detections = detections[detections[:, 4] >= self.conf_threshold]
                
                # Apply NMS (Non-Maximum Suppression)
                if len(detections) > 0:
                    detections = self._nms(detections, iou_threshold=0.45)
                
                return detections.astype(np.float32)
        finally:
            # Pop the context if we pushed it
            if context_pushed:
                try:
                    self.cuda.Context.pop()
                except:
                    pass  # Ignore errors during cleanup
    
    def _nms(self, boxes: np.ndarray, iou_threshold: float = 0.45) -> np.ndarray:
        """
        Apply Non-Maximum Suppression to remove overlapping detections.
        
        Args:
            boxes: Detection boxes [N, 6] with [x1, y1, x2, y2, conf, cls]
            iou_threshold: IoU threshold for NMS
            
        Returns:
            Filtered boxes after NMS
        """
        if len(boxes) == 0:
            return boxes
        
        # Sort by confidence (descending)
        order = boxes[:, 4].argsort()[::-1]
        boxes = boxes[order]
        
        keep = []
        while len(boxes) > 0:
            # Keep the box with highest confidence
            keep.append(boxes[0])
            
            if len(boxes) == 1:
                break
            
            # Compute IoU with remaining boxes
            ious = self._compute_iou_batch(boxes[0:1], boxes[1:])
            
            # Remove boxes with IoU > threshold
            mask = ious < iou_threshold
            boxes = boxes[1:][mask]
        
        return np.array(keep) if keep else np.array([]).reshape(0, 6)
    
    def _compute_iou_batch(self, box1: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Compute IoU between one box and multiple boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1[0, 0], box1[0, 1], box1[0, 2], box1[0, 3]
        x1_2, y1_2, x2_2, y2_2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        
        # Intersection
        x1_i = np.maximum(x1_1, x1_2)
        y1_i = np.maximum(y1_1, y1_2)
        x2_i = np.minimum(x2_1, x2_2)
        y2_i = np.minimum(y2_1, y2_2)
        
        inter_area = np.maximum(0, x2_i - x1_i) * np.maximum(0, y2_i - y1_i)
        
        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area
        
        # IoU
        iou = inter_area / (union_area + 1e-6)
        return iou
    
    def postprocess(self, detections: np.ndarray, original_shape: Tuple[int, int]) -> List[Dict]:
        """
        Post-process detection results and scale to original image size.
        
        Args:
            detections: Raw detection output [N, 6] with [x1, y1, x2, y2, conf, cls]
            original_shape: Original image shape (height, width)
            
        Returns:
            List of detection dicts with keys: bbox, cls, conf
        """
        results = []
        orig_h, orig_w = original_shape
        scale_x = orig_w / self.input_size[0]
        scale_y = orig_h / self.input_size[1]
        
        for det in detections:
            if det[4] < self.conf_threshold:
                continue
            
            x1, y1, x2, y2 = det[0:4]
            # Scale to original image size
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)
            
            # Clip to image bounds
            x1 = max(0, min(x1, orig_w - 1))
            y1 = max(0, min(y1, orig_h - 1))
            x2 = max(0, min(x2, orig_w - 1))
            y2 = max(0, min(y2, orig_h - 1))
            
            results.append({
                'bbox': [x1, y1, x2, y2],
                'cls': int(det[5]),
                'conf': float(det[4])
            })
        
        return results
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Run detection on an image.
        
        Args:
            image: BGR image (H, W, 3)
            
        Returns:
            List of detection dicts with keys: bbox, cls, conf
        """
        original_shape = image.shape[:2]
        preprocessed = self.preprocess(image)
        raw_detections = self.infer(preprocessed)
        detections = self.postprocess(raw_detections, original_shape)
        return detections

