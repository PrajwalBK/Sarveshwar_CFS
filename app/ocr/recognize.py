"""
TensorRT OCR Recognizer (CRNN/PP-OCR)

This module provides OCR recognition using TensorRT CRNN/PP-OCR engines.
"""

import numpy as np
import cv2
from typing import Dict, Tuple, List
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


class PlateRecognizer:
    """
    TensorRT OCR recognizer wrapper.
    
    Supports CRNN/PP-OCR-style CTC-based text recognition.
    """
    
    def __init__(self, engine_path: str, alphabet_path: str, input_size: Tuple[int, int] = (320, 32)):
        """
        Initialize TensorRT OCR recognizer.
        
        Args:
            engine_path: Path to TensorRT OCR engine file (.engine)
            alphabet_path: Path to alphabet.txt file
            input_size: Input image size (width, height) for OCR
        """
        self.engine_path = engine_path
        self.input_size = input_size
        
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"TensorRT OCR engine not found: {engine_path}")
        
        # Load alphabet - PaddleOCR uses one character per line
        # Index 0 = blank token, Index 1+ = characters from file
        with open(alphabet_path, 'r', encoding='utf-8') as f:
            lines = f.read().strip().split('\n')
            # Remove empty lines
            self.alphabet_list = [line for line in lines if line.strip()]
        
        # CTC blank is at index 0
        self.blank_idx = 0
        self.alphabet_size = len(self.alphabet_list)
        
        # For backwards compatibility, also store as string
        self.alphabet = ''.join(self.alphabet_list)
        
        # Note: PaddleOCR models with 6625 output classes use a larger dictionary
        # (e.g., ppocr_keys_v1.txt). If the model outputs indices > alphabet_size,
        # those are likely special tokens or characters not in our alphabet.
        # For now, we only decode indices 1 to alphabet_size.
        
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT or PyCUDA not available. Cannot initialize OCR.")
        
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
            device = cuda.Device(0)
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
        
        print(f"Loaded TensorRT OCR engine: {engine_path}")
        print(f"  Input tensor: {self.input_names[0]}, shape: {self.input_shape}")
        print(f"  Output tensor: {self.output_names[0]}, shape: {self.output_shape}")
        if self.alphabet_size <= 100:
            preview = ''.join(self.alphabet_list)
        else:
            preview = ''.join(self.alphabet_list[:50]) + "..."
        print(f"Alphabet: {alphabet_path} ({self.alphabet_size} chars)")
        print(f"Alphabet preview: {preview}")
    
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
                
                # Handle dynamic shapes and zero dimensions
                if size <= 0:  # Dynamic shape or zero dimension
                    # For OCR output: shape is typically (batch, sequence_length, num_classes)
                    # If sequence_length is 0, estimate a reasonable max sequence length
                    if len(tensor_shape) == 3 and tensor_shape[1] <= 0:
                        # OCR output: (1, 0, 97) or (1, -1, 97) -> estimate max 200 characters
                        max_seq_len = 200
                        size = tensor_shape[0] * max_seq_len * tensor_shape[2]
                        # Create concrete shape for buffer allocation
                        buffer_shape = (tensor_shape[0], max_seq_len, tensor_shape[2])
                        print(f"  Warning: Dynamic output shape {tensor_shape}, estimating size for {max_seq_len} max sequence length")
                    elif len(tensor_shape) == 4 and tensor_shape[3] == -1:
                        # OCR input: (1, 3, 48, -1) -> set width to 320
                        width = 320
                        size = tensor_shape[0] * tensor_shape[1] * tensor_shape[2] * width
                        buffer_shape = (tensor_shape[0], tensor_shape[1], tensor_shape[2], width)
                        print(f"  Warning: Dynamic input shape {tensor_shape}, setting width to {width}")
                    elif size < 0:  # Negative means fully dynamic
                        # Estimate based on other dimensions
                        if len(tensor_shape) >= 2:
                            # Assume batch dimension is 1, estimate from other dimensions
                            estimated_dims = [s if s > 0 else 100 for s in tensor_shape]
                            buffer_shape = tuple(estimated_dims)
                            size = np.prod(estimated_dims)
                        else:
                            buffer_shape = (1024,)
                            size = 1024
                    else:  # size == 0
                        # Zero dimension - use reasonable default
                        if len(tensor_shape) == 3:
                            # Assume it's OCR output: estimate 200 sequence length
                            max_seq_len = 200
                            size = tensor_shape[0] * max_seq_len * tensor_shape[2]
                            buffer_shape = (tensor_shape[0], max_seq_len, tensor_shape[2])
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
                print(f"Warning: Failed to process tensor {i} ({tensor_name if 'tensor_name' in locals() else 'unknown'}): {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if len(inputs) == 0:
            raise RuntimeError("No input tensors found in engine")
        if len(outputs) == 0:
            raise RuntimeError("No output tensors found in engine")
        
        return inputs, outputs, input_names, output_names, bindings, stream
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess cropped image for OCR input.
        
        PaddleOCR recognition models expect:
        - RGB input (3 channels)
        - Normalization: img / 255.0 (normalized to [0, 1])
        - Height: 48 pixels (for PaddleOCR models)
        
        Note: Some PaddleOCR models use (img - 127.5) / 127.5, but testing shows
        [0, 1] normalization produces better results for this model.
        
        Args:
            image: BGR cropped image (H, W, 3) or grayscale (H, W)
            
        Returns:
            Preprocessed image tensor (1, 3, H, W) normalized to [0, 1]
        """
        # Improve upscaling for small text - use better interpolation
        # If image is very small, upscale more aggressively before resizing to model input
        h, w = image.shape[:2]
        target_h = self.input_size[1]  # 48 for PaddleOCR
        
        # If input is much smaller than target, upscale first with better interpolation
        if h < target_h * 0.8:  # If height is less than 80% of target
            # Upscale to at least 2x target height for better quality
            scale = max(2.0, target_h * 2.0 / h)
            new_h = int(h * scale)
            new_w = int(w * scale)
            # Use cubic interpolation for better quality on upscaling
            img_upscaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        else:
            img_upscaled = image
        
        # Resize to input size (width, height) - use area interpolation for downscaling
        # Area interpolation is better for downscaling, linear is fine for upscaling
        if img_upscaled.shape[0] > target_h or img_upscaled.shape[1] > self.input_size[0]:
            # Downscaling - use area interpolation
            img_resized = cv2.resize(img_upscaled, self.input_size, interpolation=cv2.INTER_AREA)
        else:
            # Upscaling or same size - use cubic for better quality
            img_resized = cv2.resize(img_upscaled, self.input_size, interpolation=cv2.INTER_CUBIC)
        
        # Convert BGR to RGB if needed (PaddleOCR expects RGB)
        if len(img_resized.shape) == 3 and img_resized.shape[2] == 3:
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        elif len(img_resized.shape) == 2:
            # Grayscale - convert to RGB by replicating channels
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = img_resized
        
        # PaddleOCR recognition models: Use [0, 1] normalization
        # Many PaddleOCR models are trained with simple [0, 1] normalization
        # ImageNet normalization was causing garbled text (e.g., "PMARPONS" instead of "MAR JON")
        img_float = img_rgb.astype(np.float32)
        img_normalized = img_float / 255.0
        
        # Add batch dimension and transpose to CHW format: (1, 3, H, W)
        # PaddleOCR expects NCHW format
        img_batched = np.expand_dims(img_normalized.transpose(2, 0, 1), axis=0)
        
        return img_batched
    
    def infer(self, preprocessed: np.ndarray) -> np.ndarray:
        """
        Run TensorRT OCR inference.
        
        Args:
            preprocessed: Preprocessed image tensor (1, 1, H, W)
            
        Returns:
            Character probability logits: [T, C] where T is sequence length, C is alphabet size
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

                    
                    if main_module is not None:
                        primary_ctx = main_module._cuda_primary_context
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
                import pycuda.autoinit
                current_ctx = self.cuda.Context.get_current()
                if current_ctx is None:
                    raise RuntimeError(f"Failed to initialize CUDA context: {ctx_err}")
            except Exception as e2:
                raise RuntimeError(f"Failed to activate CUDA context. Error: {ctx_err}, {e2}")
        
        try:
            with self._inference_lock:
                if len(self.inputs) == 0:
                    raise RuntimeError("No input tensors available")
                
                input_tensor = self.inputs[0]
                input_name = input_tensor['name']
                
                # For dynamic shapes, set the concrete input shape in the context FIRST
                # This is critical for TensorRT engines with dynamic dimensions
                # We need to set the shape before copying data, as it may affect buffer requirements
                original_shape = input_tensor['shape']
                expected_shape = input_tensor.get('buffer_shape', input_tensor['shape'])
                
                if any(s == -1 or s < 0 for s in original_shape):
                    # Has dynamic dimensions - set concrete shape BEFORE preparing data
                    concrete_input_shape = expected_shape  # Use the buffer_shape we calculated
                    try:
                        # Try TensorRT 10.x API
                        if hasattr(self.context, 'set_input_shape'):
                            self.context.set_input_shape(self.input_names[0], concrete_input_shape)
                        elif hasattr(self.context, 'set_tensor_shape'):
                            self.context.set_tensor_shape(self.input_names[0], concrete_input_shape)
                        # After setting input shape, output shape may have changed - get actual shape
                        if len(self.output_names) > 0:
                            try:
                                actual_output_shape_dims = self.context.get_tensor_shape(self.output_names[0])
                                output_tensor = self.outputs[0]
                                
                                # Convert Dims to tuple for easier handling
                                if hasattr(actual_output_shape_dims, '__iter__'):
                                    actual_output_shape = tuple(actual_output_shape_dims)
                                else:
                                    # Fallback: try to access as tuple
                                    actual_output_shape = tuple([actual_output_shape_dims[i] for i in range(len(actual_output_shape_dims))])
                                
                                # Verify output buffer is large enough
                                actual_output_size = trt.volume(actual_output_shape_dims)
                                if actual_output_size < 0:
                                    # Dynamic output - estimate size
                                    if len(actual_output_shape) == 3 and actual_output_shape[1] <= 0:
                                        # OCR output: estimate max sequence length
                                        max_seq_len = 200
                                        actual_output_size = actual_output_shape[0] * max_seq_len * actual_output_shape[2]
                                    else:
                                        # Fallback estimation
                                        actual_output_size = abs(actual_output_size) * 100
                                
                                buffer_output_size = np.prod(output_tensor['buffer_shape'])
                                if actual_output_size > buffer_output_size:
                                    print(f"  Error: Output shape {actual_output_shape} requires {actual_output_size} elements, but buffer only has {buffer_output_size}")
                                    raise RuntimeError(f"Output buffer too small for dynamic output shape")
                                
                                # Update the output tensor's buffer_shape to match actual shape
                                # This ensures we're using the correct shape for the rest of inference
                                # Compare as tuples to avoid Dims comparison issues
                                buffer_shape_tuple = tuple(output_tensor['buffer_shape'])
                                if actual_output_shape != buffer_shape_tuple:
                                    # Shape changed - update it (but keep the same buffer, it should be large enough)
                                    output_tensor['buffer_shape'] = actual_output_shape
                                    print(f"  Info: Output shape updated to {actual_output_shape} after setting input shape")
                            except Exception as out_shape_err:
                                # If we can't verify, continue but log warning
                                print(f"  Warning: Could not verify output shape: {out_shape_err}")
                                pass
                    except Exception as shape_err:
                        print(f"  Warning: Could not set dynamic input shape: {shape_err}")
                        # Continue anyway - might still work
                
                # Prepare input data - ensure it matches the expected shape
                # Preprocessed is now (1, 3, H, W) - RGB format with PaddleOCR normalization
                input_data = preprocessed.astype(input_tensor['dtype'])
                
                # Ensure data is contiguous in memory (required for CUDA)
                if not input_data.flags['C_CONTIGUOUS']:
                    input_data = np.ascontiguousarray(input_data)
                
                # Handle shape conversion: preprocessed is (1, 3, H, W)
                # Get expected height from engine shape (48 for PaddleOCR)
                expected_height = expected_shape[2] if len(expected_shape) >= 3 else 48
                
                # Resize height if needed (width is dynamic, height should match expected)
                if len(input_data.shape) == 4 and input_data.shape[1] == 3:
                    h, w = input_data.shape[2], input_data.shape[3]
                    # Resize to expected height (from engine)
                    if h != expected_height:
                        import cv2
                        # Resize each channel separately to maintain RGB format
                        img_resized_channels = []
                        for c in range(3):
                            channel_2d = input_data[0, c, :, :]  # (H, W)
                            channel_resized = cv2.resize(channel_2d, (w, expected_height), interpolation=cv2.INTER_LINEAR)  # (expected_height, W)
                            img_resized_channels.append(channel_resized)
                        # Stack channels: (3, expected_height, W)
                        img_resized = np.stack(img_resized_channels, axis=0)
                        # Add batch dimension: (1, 3, expected_height, W)
                        input_data = np.expand_dims(img_resized, axis=0)
                        # Ensure contiguous after reshape
                        if not input_data.flags['C_CONTIGUOUS']:
                            input_data = np.ascontiguousarray(input_data)
                
                # Reshape to match the host buffer shape
                # Preprocessed is (1, 3, H, W), engine expects (1, 3, 48, W) or similar
                if input_data.shape != expected_shape:
                    # Check if we can reshape (size matches but shape differs)
                    if input_data.size == np.prod(expected_shape):
                        input_data = input_data.reshape(expected_shape)
                    elif len(expected_shape) == 4 and expected_shape[1] == 3:
                        # Engine expects RGB (3 channels) - ensure we have 3 channels
                        if input_data.shape[1] != 3:
                            # Replicate channel if needed (shouldn't happen with new preprocessing)
                            if input_data.shape[1] == 1:
                                input_data = np.repeat(input_data, 3, axis=1)
                        # Handle height/width mismatch
                        if expected_shape[2] != input_data.shape[2] or expected_shape[3] != input_data.shape[3]:
                            import cv2
                            h_actual, w_actual = input_data.shape[2], input_data.shape[3]
                            h_expected, w_expected = expected_shape[2], expected_shape[3]
                            # Resize each channel
                            img_resized_channels = []
                            for c in range(3):
                                channel_2d = input_data[0, c, :, :]  # (H, W)
                                channel_resized = cv2.resize(channel_2d, (w_expected, h_expected), interpolation=cv2.INTER_LINEAR)
                                img_resized_channels.append(channel_resized)
                            img_resized = np.stack(img_resized_channels, axis=0)
                            input_data = np.expand_dims(img_resized, axis=0)
                    
                    # Check if we can reshape
                    if input_data.shape == expected_shape:
                        pass  # Already correct shape
                    elif input_data.size == np.prod(expected_shape):
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
                
                # Validate input data size matches buffer size
                input_size = input_data.size
                buffer_size = input_tensor['host'].size
                if input_size != buffer_size:
                    # Try to reshape if total size matches
                    if input_size == buffer_size:
                        input_data = input_data.reshape(input_tensor['host'].shape)
                    else:
                        raise RuntimeError(
                            f"Input data size mismatch: input has {input_size} elements, "
                            f"but buffer expects {buffer_size} elements. "
                            f"Input shape: {input_data.shape}, Expected shape: {expected_shape}"
                        )
                
                # Copy to host buffer (host buffer has the same shape)
                # Ensure input_data is the right shape before copying
                if input_data.shape != input_tensor['host'].shape:
                    input_data = input_data.reshape(input_tensor['host'].shape)
                
                np.copyto(input_tensor['host'], input_data)
                
                # Debug: Log input statistics occasionally to verify preprocessing
                import random
                if random.random() < 0.01:  # 1% chance
                    print(f"[OCR] Input debug: shape={input_data.shape}, dtype={input_data.dtype}, "
                          f"min={np.min(input_data):.4f}, max={np.max(input_data):.4f}, "
                          f"mean={np.mean(input_data):.4f}, std={np.std(input_data):.4f}")
                
                # Validate device memory pointers before setting addresses
                input_device_ptr = int(input_tensor['device'])
                if input_device_ptr == 0:
                    raise RuntimeError("Invalid input device memory pointer (null)")
                
                output_device_ptr = None
                if len(self.output_names) > 0:
                    output_device_ptr = int(self.outputs[0]['device'])
                    if output_device_ptr == 0:
                        raise RuntimeError("Invalid output device memory pointer (null)")
                
                # IMPORTANT: Set tensor addresses AFTER setting input shape (if dynamic)
                # This ensures TensorRT knows the correct shapes before we set addresses
                # For dynamic shapes, the addresses might need to be re-set after shape changes
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
                        # Print debug info for OCR inference
                        # Suppress Cask errors if they're non-fatal (they seem to be warnings in some TensorRT versions)
                        import warnings
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=RuntimeWarning)
                            self.context.execute_async_v3(stream_handle=inference_stream.handle)
                    except (RuntimeError, Exception) as e:
                        error_str = str(e).lower()
                        error_msg = f"TensorRT inference failed: {e}"
                        
                        # Check if it's a resource handle error or Cask error
                        if "invalid resource handle" in error_str or "cuda" in error_str or "cutensor" in error_str or "cask" in error_str:
                            print(f"  CUDA resource error detected. Attempting recovery...")
                            print(f"  Input shape: {expected_shape}, Input device ptr: {input_device_ptr}")
                            if len(self.output_names) > 0:
                                print(f"  Output shape: {self.outputs[0]['buffer_shape']}, Output device ptr: {output_device_ptr}")
                            
                            # Try to recover by re-setting everything
                            try:
                                # Re-set input shape if dynamic
                                if any(s == -1 or s < 0 for s in original_shape):
                                    try:
                                        if hasattr(self.context, 'set_input_shape'):
                                            self.context.set_input_shape(self.input_names[0], expected_shape)
                                        elif hasattr(self.context, 'set_tensor_shape'):
                                            self.context.set_tensor_shape(self.input_names[0], expected_shape)
                                    except:
                                        pass
                                
                                # Re-set tensor addresses
                                self.context.set_tensor_address(self.input_names[0], input_device_ptr)
                                if len(self.output_names) > 0:
                                    self.context.set_tensor_address(self.output_names[0], output_device_ptr)
                                
                                # Retry inference once
                                print(f"  Retrying inference...")
                                self.context.execute_async_v3(stream_handle=inference_stream.handle)
                                print(f"  Retry succeeded!")
                            except Exception as e2:
                                # Provide specific guidance for Cask errors
                                cask_guidance = ""
                                if "cask" in error_str:
                                    cask_guidance = (
                                        "\n  Cask error specific guidance:\n"
                                        "  - Verify input shape matches engine expectations exactly\n"
                                        "  - Ensure input data is C-contiguous (use np.ascontiguousarray)\n"
                                        "  - Check that buffer sizes match data sizes\n"
                                        "  - Try rebuilding the engine with --fp32 instead of --fp16\n"
                                        "  - Verify GPU memory is not corrupted\n"
                                    )
                                
                                raise RuntimeError(
                                    f"CUDA inference failed after recovery attempt.\n"
                                    f"Original error: {e}\n"
                                    f"Recovery error: {e2}\n"
                                    f"This may indicate:\n"
                                    f"  1. Engine was built on a different device\n"
                                    f"  2. CUDA context is invalid\n"
                                    f"  3. Dynamic shape handling issue\n"
                                    f"  4. Insufficient GPU memory\n"
                                    f"  5. Input data shape/size mismatch{cask_guidance}"
                                )
                        else:
                            # Non-resource error - raise as-is
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
                
                # Use buffer_shape (actual shape after dynamic resolution) if available, otherwise use original shape
                output_shape = output_tensor.get('buffer_shape', output_tensor['shape'])
                
                # Get the actual output shape from TensorRT context (most accurate)
                try:
                    if len(self.output_names) > 0:
                        actual_output_shape_dims = self.context.get_tensor_shape(self.output_names[0])
                        # Convert Dims to tuple for easier handling
                        if hasattr(actual_output_shape_dims, '__iter__'):
                            actual_output_shape = tuple(actual_output_shape_dims)
                        else:
                            # Fallback: try to access as tuple
                            actual_output_shape = tuple([actual_output_shape_dims[i] for i in range(len(actual_output_shape_dims))])
                        
                        if actual_output_shape and all(s > 0 for s in actual_output_shape):
                            output_shape = actual_output_shape
                except Exception as shape_err:
                    pass
                    # Fall back to buffer_shape
                
                # Use .size for numpy arrays (total number of elements), not len() which returns first dimension
                output_size = output.size if hasattr(output, 'size') else len(output)
                
                # Ensure output_shape is a tuple/list for consistent handling
                if not isinstance(output_shape, (tuple, list)):
                    output_shape = tuple(output_shape) if hasattr(output_shape, '__iter__') else (output_shape,)
                
                # Reshape output based on actual output shape
                # CRNN/PP-OCR output: [batch, sequence_length, num_classes] or [sequence_length, num_classes]
                # Use .size for numpy arrays (total number of elements), not len() which returns first dimension
                total_elements = output.size if hasattr(output, 'size') else len(output)
                
                if len(output_shape) == 3:
                    # [batch, T, C] or [T, batch, C] - TensorRT might return different order
                    # Check if dimensions are swapped: if first dim is large and second is 1, likely swapped
                    # Model outputs 6625 classes (blank + 6623 chars + 1 padding token)
                    num_classes_from_shape = output_shape[2]
                    if output_shape[0] > 1 and output_shape[1] == 1 and num_classes_from_shape >= self.alphabet_size:
                        # Likely swapped: (seq_len, 1, num_classes) -> should be (1, seq_len, num_classes)
                        batch_size = 1
                        seq_len = output_shape[0]
                        num_classes = output_shape[2]
                        # Reshape as (seq_len, batch, num_classes) then transpose
                        expected_size = seq_len * batch_size * num_classes
                        if expected_size > 0 and expected_size <= total_elements:
                            output_flat = output.flatten()[:expected_size]
                            output = output_flat.reshape(seq_len, batch_size, num_classes)
                            # Transpose from (seq_len, batch, num_classes) to (batch, seq_len, num_classes)
                            output = np.transpose(output, (1, 0, 2))  # -> (batch, seq_len, num_classes)
                            output = output[0]  # Remove batch dimension -> [seq_len, num_classes]
                            # Debug: log when we detect swapped dimensions (only once per session)
                            if not hasattr(self, '_swapped_dim_logged'):
                                print(f"[OCR] Detected swapped dimensions: {output_shape} -> (1, {seq_len}, {num_classes}), final shape: {output.shape}")
                                self._swapped_dim_logged = True
                        else:
                            # Fallback: treat as (seq_len, num_classes) directly
                            seq_len = total_elements // num_classes
                            if seq_len > 0:
                                output_flat = output.flatten()[:seq_len * num_classes]
                                output = output_flat.reshape(seq_len, num_classes)
                            else:
                                output_flat = output.flatten()[:num_classes]
                                output = output_flat.reshape(1, num_classes)
                    else:
                        # Normal case: [batch, T, C]
                        batch_size = output_shape[0]
                        seq_len = output_shape[1]
                        num_classes = output_shape[2]
                        
                        # Calculate expected size
                        expected_size = batch_size * seq_len * num_classes
                        
                        if expected_size > 0 and expected_size <= total_elements:
                            # Reshape to [batch, seq_len, num_classes]
                            # Flatten first to ensure we have a 1D array, then reshape
                            output_flat = output.flatten()[:expected_size]
                            output = output_flat.reshape(batch_size, seq_len, num_classes)
                            output = output[0]  # Remove batch dimension -> [seq_len, num_classes]
                        else:
                            # Fallback: calculate from buffer size
                            actual_seq_len = total_elements // (batch_size * num_classes)
                            if actual_seq_len > 0:
                                expected_size = batch_size * actual_seq_len * num_classes
                                output_flat = output.flatten()[:expected_size]
                                output = output_flat.reshape(batch_size, actual_seq_len, num_classes)
                                output = output[0]  # Remove batch dimension -> [T, C]
                            else:
                                # Last resort: try to infer from total elements
                                T = total_elements // num_classes
                                if T > 0:
                                    output_flat = output.flatten()[:T * num_classes]
                                    output = output_flat.reshape(T, num_classes)
                                else:
                                    output_flat = output.flatten()[:num_classes]
                                    output = output_flat.reshape(1, num_classes)
                elif len(output_shape) == 2:
                    # [T, C]
                    expected_size = output_shape[0] * output_shape[1]
                    if expected_size > 0 and expected_size <= total_elements:
                        output_flat = output.flatten()[:expected_size]
                        output = output_flat.reshape(output_shape)
                    else:
                        # Fallback
                        if output_shape[0] > 0 and output_shape[1] > 0:
                            output_flat = output.flatten()[:expected_size]
                            output = output_flat.reshape(output_shape[0], output_shape[1])
                        else:
                            output_flat = output.flatten()[:output_shape[1]]
                            output = output_flat.reshape(1, output_shape[1])
                else:
                    # Fallback: Flatten and reshape to [T, C]
                    # Try to infer num_classes from output_shape if available, otherwise use a reasonable default
                    # PaddleOCR models output 6625 classes (blank + 6623 chars + 1 padding)
                    if len(output_shape) > 0 and output_shape[-1] > 0:
                        num_classes = output_shape[-1]
                    else:
                        # Default to model's expected output size (6625 for PaddleOCR)
                        # This is larger than alphabet_size (6623) to account for blank and padding tokens
                        num_classes = max(self.alphabet_size + 2, 6625)  # blank + chars + padding
                    
                    T = total_elements // num_classes
                    if T > 0:
                        output_flat = output.flatten()[:T * num_classes]
                        output = output_flat.reshape(T, num_classes)
                    else:
                        output_flat = output.flatten()[:num_classes]
                        output = output_flat.reshape(1, num_classes)
                
                # Apply softmax to get probabilities (if output is logits)
                # Many models output logits, so we apply softmax
                # If your model already outputs probabilities, you can skip this
                exp_output = np.exp(output - np.max(output, axis=1, keepdims=True))
                probs = exp_output / (np.sum(exp_output, axis=1, keepdims=True) + 1e-8)
                
                # Debug: Log output shape occasionally
                import random
                if random.random() < 0.01:  # 1% chance
                    print(f"[OCR] Output shape after processing: {probs.shape}, max prob: {np.max(probs):.4f}, "
                          f"non-blank predictions: {np.sum(np.argmax(probs, axis=1) != 0)}/{probs.shape[0]}")
                
                return probs.astype(np.float32)
        finally:
            # Pop the context if we pushed it
            if context_pushed:
                try:
                    self.cuda.Context.pop()
                except:
                    pass  # Ignore errors during cleanup
    
    def ctc_decode(self, logits: np.ndarray) -> Tuple[str, float]:
        """
        Decode CTC output to text string using greedy decoding.
        
        Args:
            logits: Character probability logits [T, C] (already softmaxed)
            
        Returns:
            Tuple of (recognized_text, confidence)
        """
        if logits.shape[0] == 0:
            return "", 0.0
        
        # Greedy CTC decoding
        # Find most likely character at each timestep
        char_indices = np.argmax(logits, axis=1)
        probs = np.max(logits, axis=1)
        
        # Remove blanks and duplicates (CTC decoding)
        decoded_chars = []
        decoded_probs = []
        prev_idx = self.blank_idx
        
        # PaddleOCR model outputs indices 0-6624 where:
        # - Index 0 = blank token (skip)
        # - Index 1 = first character in dictionary (alphabet_list[0])
        # - Index 2 = second character (alphabet_list[1]), etc.
        # - Index 6624 = last class (might be padding/end token)
        
        for idx, prob in zip(char_indices, probs):
            # Skip blank tokens (index 0)
            if idx == self.blank_idx:
                prev_idx = idx
                continue
            
            # Skip duplicate consecutive characters (CTC collapse)
            if idx != prev_idx:
                # Map index to alphabet character
                # Index 1 -> alphabet_list[0], index 2 -> alphabet_list[1], etc.
                if idx > 0 and idx <= self.alphabet_size:
                    decoded_chars.append(self.alphabet_list[idx - 1])
                    decoded_probs.append(prob)
                # If index is beyond alphabet size, it might be a special token
                # For PaddleOCR models with 6625 classes, index 6624 might be padding
                # Log occasionally for debugging
                elif idx > self.alphabet_size:
                    import random
                    if random.random() < 0.01:  # 1% chance
                        print(f"[OCR] Warning: Model output index {idx} exceeds alphabet size {self.alphabet_size} (max valid: {self.alphabet_size})")
            
            prev_idx = idx
        
        text = ''.join(decoded_chars)
        
        # Filter to only English letters, numbers, and common symbols
        # This removes Chinese characters and other non-English text
        import re
        # Allow: A-Z, a-z, 0-9, space, period, hyphen, underscore
        filtered_text = re.sub(r'[^A-Za-z0-9 .\-_]', '', text)
        
        # Calculate confidence for multi-class CTC models
        # With 6625 classes, raw probabilities are naturally very low (e.g., 0.0004)
        # Use a confidence metric that accounts for the large number of classes
        if len(decoded_probs) > 0:
            # Method 1: Scale by expected random probability
            # For 6625 classes, random chance = 1/6625 ≈ 0.00015
            # If our average prob is 0.0004, that's about 2.67x better than random
            random_prob = 1.0 / 6625  # Expected probability for random guess
            avg_prob = float(np.mean(decoded_probs))
            
            # Calculate how much better than random (ratio)
            # Then normalize to 0-1 range: ratio of 1.0 = random, ratio of 10 = very confident
            ratio = avg_prob / random_prob if random_prob > 0 else 0.0
            # Normalize: ratio of 1 = 0.0 confidence, ratio of 10+ = 1.0 confidence
            # Use a sigmoid-like function: confidence = 1 - exp(-ratio/5)
            confidence = float(1.0 - np.exp(-ratio / 5.0))
            
            # Penalize confidence if we filtered out many characters (indicates wrong predictions)
            if len(filtered_text) < len(text) * 0.5:  # If we filtered out more than 50%
                confidence *= 0.5  # Reduce confidence significantly
            
            # Ensure confidence is in valid range
            confidence = max(0.0, min(1.0, confidence))
        else:
            confidence = 0.0
        
        # Return filtered text (only English/numbers)
        return filtered_text, confidence
    
    def recognize(self, image: np.ndarray) -> Dict[str, any]:
        """
        Recognize text in a cropped image region.
        
        Args:
            image: BGR cropped image (H, W, 3)
            
        Returns:
            Dict with keys: text, conf
        """
        preprocessed = self.preprocess(image)
        logits = self.infer(preprocessed)
        
        # Debug: Check if model is producing any non-blank outputs
        if logits.shape[0] > 0:
            # Get max probability indices for each timestep
            max_indices = np.argmax(logits, axis=1)
            max_probs = np.max(logits, axis=1)
            # Count non-blank predictions
            non_blank_count = np.sum(max_indices != self.blank_idx)
            # Check if any predictions have reasonable confidence
            high_conf_count = np.sum(max_probs > 0.1)
            
            # Only log if we're getting unexpected results (all blanks or very low confidence)
            if non_blank_count == 0 or high_conf_count == 0:
                # This is likely a model issue - log once per 100 calls to avoid spam
                import random
                if random.random() < 0.01:  # 1% chance to log
                    print(f"[OCR] Debug: logits shape={logits.shape}, non-blank={non_blank_count}/{len(max_indices)}, "
                          f"high-conf={high_conf_count}/{len(max_indices)}, "
                          f"max_prob={np.max(max_probs):.4f}, sample_indices={max_indices[:10].tolist()}")
        
        text, conf = self.ctc_decode(logits)
        
        # Additional filtering: reject results that are too short or have very low confidence
        # For trailer IDs, we expect at least 2 characters
        MIN_TEXT_LENGTH = 2
        MIN_CONFIDENCE = 0.15  # Reject very low confidence predictions
        
        # Reject obviously wrong results:
        # - Single characters (unless they're numbers with high confidence)
        # - Very short text with low confidence
        # - Text that's mostly special characters
        text_stripped = text.strip()
        if len(text_stripped) == 0:
            return {'text': '', 'conf': 0.0}
        
        # Allow single characters only if they're digits with decent confidence
        if len(text_stripped) == 1:
            if text_stripped.isdigit() and conf >= 0.25:
                # Single digit with good confidence - might be part of a number
                pass  # Allow it
            else:
                # Single non-digit or low confidence - reject
                return {'text': '', 'conf': 0.0}
        
        # Reject if confidence is too low
        if conf < MIN_CONFIDENCE:
            return {'text': '', 'conf': 0.0}
        
        # Reject if text is too short (unless it's a number)
        if len(text_stripped) < MIN_TEXT_LENGTH:
            if not text_stripped.isdigit():
                return {'text': '', 'conf': 0.0}
        
        return {
            'text': text.strip(),
            'conf': conf
        }

