"""
TensorRT TrOCR Recognizer

TrOCR (Transformer-based OCR) recognizer using TensorRT engine.
TrOCR uses a VisionEncoderDecoder architecture with BERT tokenizer for decoding.
"""

import os
import numpy as np
import cv2
from typing import Dict, Tuple, Optional
import sys
import threading

# Try to import TensorRT and PyCUDA
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError as e:
    TRT_AVAILABLE = False
    print(f"Warning: TensorRT or PyCUDA not available: {e}")

# Try to import transformers for tokenizer
try:
    from transformers import TrOCRProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. TrOCR decoding may be limited.")


class TrOCRRecognizer:
    """
    TensorRT TrOCR recognizer wrapper.
    
    Supports TrOCR models converted to TensorRT.
    Uses transformers tokenizer for text decoding.
    """
    
    @staticmethod
    def _dims_to_tuple(dims):
        """Convert TensorRT Dims object to tuple."""
        try:
            return tuple(dims) if dims else ()
        except:
            # If direct conversion fails, iterate through Dims
            return tuple([d for d in dims])
    
    def __init__(self, engine_path: str, model_dir: Optional[str] = None, input_size: Tuple[int, int] = (384, 384)):
        """
        Initialize TensorRT TrOCR recognizer.
        
        Args:
            engine_path: Path to TensorRT TrOCR engine file (.engine)
            model_dir: Optional path to TrOCR model directory (for tokenizer)
                      If None, tries to find tokenizer in standard locations
            input_size: Input image size (width, height) for TrOCR (default: 384x384)
        """
        self.engine_path = engine_path
        self.input_size = input_size
        
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"TensorRT TrOCR engine not found: {engine_path}")
        
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT or PyCUDA not available. Cannot initialize TrOCR.")
        
        # Initialize TensorRT runtime
        self.trt = trt
        self.cuda = cuda
        
        # Store primary CUDA context for thread safety
        # The primary context is created by pycuda.autoinit in the main thread
        # For multi-threaded use, we'll ensure each thread has a valid context
        # Use retain_primary_context() to get the context created by autoinit
        try:
            # Try to get current context first
            self.primary_context = cuda.Context.get_current()
            if self.primary_context is None:
                # If no context exists, try to retain primary context from device
                try:
                    device = cuda.Device(0)
                    self.primary_context = device.retain_primary_context()
                    if self.primary_context is not None:
                        # Activate it in this thread
                        self.primary_context.push()
                        self.primary_context.pop()  # Pop immediately, we'll push when needed
                except Exception as ctx_error:
                    print(f"  Warning: Could not retain primary CUDA context: {ctx_error}")
                    # Last resort: create a new context
                    try:
                        device = cuda.Device(0)
                        device.make_context()
                        self.primary_context = cuda.Context.get_current()
                    except Exception as create_error:
                        print(f"  Warning: Could not create CUDA context: {create_error}")
                        self.primary_context = None
        except Exception as e:
            print(f"  Warning: Could not get primary CUDA context: {e}")
            # Try to retain primary context
            try:
                device = cuda.Device(0)
                self.primary_context = device.retain_primary_context()
                if self.primary_context is None:
                    device.make_context()
                    self.primary_context = cuda.Context.get_current()
            except Exception as ctx_error:
                print(f"  Warning: Could not get/create CUDA context: {ctx_error}")
                self.primary_context = None
        
        # Thread lock for CUDA operations (CUDA is not thread-safe)
        self._inference_lock = threading.Lock()
        
        # Create TensorRT logger
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        
        # Load engine
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        
        runtime = trt.Runtime(TRT_LOGGER)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")
        
        # Create execution context
        self.context = self.engine.create_execution_context()
        
        # Get input/output tensor names and shapes
        self.input_names = []
        self.output_names = []
        
        # TensorRT 8.x+ uses num_io_tensors
        if hasattr(self.engine, 'num_io_tensors'):
            num_tensors = self.engine.num_io_tensors
        else:
            raise RuntimeError("TensorRT version too old. Requires TensorRT 8.x or later.")
        
        for i in range(num_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            tensor_mode = self.engine.get_tensor_mode(tensor_name)
            
            if tensor_mode == trt.TensorIOMode.INPUT:
                self.input_names.append(tensor_name)
            elif tensor_mode == trt.TensorIOMode.OUTPUT:
                self.output_names.append(tensor_name)
        
        if not self.input_names or not self.output_names:
            raise RuntimeError("Failed to get input/output tensor names from engine")
        
        # Get input shape - convert TensorRT Dims to tuple
        input_shape_raw = self.engine.get_tensor_shape(self.input_names[0])
        self.input_shape = self._dims_to_tuple(input_shape_raw)
        print(f"  Input tensor: {self.input_names[0]}, shape: {self.input_shape}")
        
        # Get output shape (may be dynamic) - convert TensorRT Dims to tuple
        output_shape_raw = self.engine.get_tensor_shape(self.output_names[0])
        self.output_shape = self._dims_to_tuple(output_shape_raw)
        print(f"  Output tensor: {self.output_names[0]}, shape: {self.output_shape}")
        
        # Allocate CUDA buffers
        self._allocate_buffers()
        
        # Load tokenizer for decoding
        self.tokenizer = None
        if TRANSFORMERS_AVAILABLE:
            self._load_tokenizer(model_dir)
        
        print(f"Loaded TensorRT TrOCR engine: {engine_path}")
    
    def _load_tokenizer(self, model_dir: Optional[str] = None):
        """Load TrOCR tokenizer for text decoding."""
        try:
            # Try to find tokenizer
            tokenizer_paths = []
            
            if model_dir and os.path.exists(model_dir):
                tokenizer_paths.append(model_dir)
            
            # Try standard locations
            tokenizer_paths.extend([
                "models/trocr_base_printed",
                "models/trocr-base-printed",
            ])
            
            # Try Hugging Face model name
            hf_model = "microsoft/trocr-base-printed"
            
            for path in tokenizer_paths:
                try:
                    if os.path.exists(path):
                        processor = TrOCRProcessor.from_pretrained(path)
                    else:
                        continue
                    self.tokenizer = processor.tokenizer
                    print(f"  Loaded tokenizer from: {path}")
                    return
                except Exception:
                    continue
            
            # Try Hugging Face directly
            try:
                processor = TrOCRProcessor.from_pretrained(hf_model)
                self.tokenizer = processor.tokenizer
                print(f"  Loaded tokenizer from Hugging Face: {hf_model}")
            except Exception as e:
                print(f"  Warning: Could not load tokenizer: {e}")
                print(f"  Text decoding will be limited")
        
        except Exception as e:
            print(f"  Warning: Tokenizer loading failed: {e}")
    
    def _allocate_buffers(self):
        """Allocate CUDA buffers for input and output."""
        self.inputs = []
        self.outputs = []
        self.bindings = []
        
        # Ensure we have a valid CUDA context before allocating buffers
        context_pushed = False
        try:
            current_ctx = cuda.Context.get_current()
            if current_ctx is None and self.primary_context is not None:
                self.primary_context.push()
                context_pushed = True
        except:
            if self.primary_context is not None:
                try:
                    self.primary_context.push()
                    context_pushed = True
                except:
                    pass
        
        try:
            # Create CUDA stream for async operations (must be in valid context)
            self.stream = cuda.Stream()
            
            for name in self.input_names:
                shape_raw = self.engine.get_tensor_shape(name)
                shape = self._dims_to_tuple(shape_raw)
                
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                size = trt.volume(shape) * np.dtype(dtype).itemsize
                
                # Allocate host and device buffers
                host_mem = cuda.pagelocked_empty(shape, dtype)
                device_mem = cuda.mem_alloc(size)
                
                self.inputs.append({'host': host_mem, 'device': device_mem, 'name': name})
            
            for name in self.output_names:
                shape_raw = self.engine.get_tensor_shape(name)
                shape = self._dims_to_tuple(shape_raw)
                
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                size = trt.volume(shape) * np.dtype(dtype).itemsize
                
                # Allocate host and device buffers
                host_mem = cuda.pagelocked_empty(shape, dtype)
                device_mem = cuda.mem_alloc(size)
                
                self.outputs.append({'host': host_mem, 'device': device_mem, 'name': name})
        finally:
            # Pop context if we pushed it
            if context_pushed:
                try:
                    cuda.Context.get_current().pop()
                except:
                    pass
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for TrOCR input.
        
        TrOCR expects:
        - RGB input (3 channels)
        - Normalized to [0, 1] then ImageNet normalization
        - Resized to input_size (default: 384x384) with aspect ratio preserved
        
        Args:
            image: BGR cropped image (H, W, 3) or grayscale (H, W)
            
        Returns:
            Preprocessed image tensor (1, 3, H, W) normalized
        """
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif len(image.shape) == 2:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = image
        
        # Preserve aspect ratio when resizing (important for text recognition)
        # TrOCR was trained on images with preserved aspect ratio
        h, w = img_rgb.shape[:2]
        target_h, target_w = self.input_size[1], self.input_size[0]  # input_size is (width, height)
        
        # Calculate scaling factor to fit image while preserving aspect ratio
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize with high-quality interpolation
        img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad to target size (TrOCR expects fixed size)
        # Pad with white background (255) which is common for text images
        pad_top = (target_h - new_h) // 2
        pad_bottom = target_h - new_h - pad_top
        pad_left = (target_w - new_w) // 2
        pad_right = target_w - new_w - pad_left
        
        img_padded = cv2.copyMakeBorder(
            img_resized, 
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, 
            value=[255, 255, 255]  # White padding
        )
        
        # Normalize: [0, 255] -> [0, 1] -> ImageNet normalization
        img_float = img_padded.astype(np.float32) / 255.0
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        img_normalized = (img_float - mean) / std
        
        # Convert to CHW format (1, 3, H, W)
        img_chw = np.transpose(img_normalized, (2, 0, 1))
        img_batch = np.expand_dims(img_chw, axis=0)
        
        return img_batch.astype(np.float32)
    
    def infer(self, preprocessed_image: np.ndarray) -> np.ndarray:
        """
        Run TrOCR inference.
        
        Args:
            preprocessed_image: Preprocessed image (1, 3, H, W)
            
        Returns:
            Model output (token logits or token IDs)
        """
        # Use lock for thread safety (CUDA operations are not thread-safe)
        with self._inference_lock:
            # Ensure CUDA context is available in this thread
            context_pushed = False
            try:
                # Try to get current context
                current_ctx = cuda.Context.get_current()
                if current_ctx is None:
                    # No context in this thread - push primary context
                    if self.primary_context is not None:
                        try:
                            self.primary_context.push()
                            context_pushed = True
                        except Exception as push_error:
                            # If push fails, try to get/create primary context
                            try:
                                # Try to get device 0's context
                                device = cuda.Device(0)
                                self.primary_context = device.retain_primary_context()
                                if self.primary_context is not None:
                                    self.primary_context.push()
                                    context_pushed = True
                                else:
                                    # Last resort: create new context (not ideal, but works)
                                    device.make_context()
                                    self.primary_context = cuda.Context.get_current()
                            except Exception as ctx_error:
                                raise RuntimeError(f"Could not establish CUDA context: {ctx_error}")
                    else:
                        # No primary context stored - try to get/create one
                        try:
                            device = cuda.Device(0)
                            self.primary_context = device.retain_primary_context()
                            if self.primary_context is not None:
                                self.primary_context.push()
                                context_pushed = True
                            else:
                                device.make_context()
                                self.primary_context = cuda.Context.get_current()
                        except Exception as ctx_error:
                            raise RuntimeError(f"Could not establish CUDA context: {ctx_error}")
            except Exception as e:
                # If we can't get current context, try to create one
                try:
                    device = cuda.Device(0)
                    device.make_context()
                    self.primary_context = cuda.Context.get_current()
                except Exception as create_error:
                    raise RuntimeError(f"Could not establish CUDA context: {create_error}")
            
            # Verify we have a valid context before proceeding
            try:
                current_ctx = cuda.Context.get_current()
                if current_ctx is None:
                    raise RuntimeError("No CUDA context available after setup")
            except Exception as ctx_check_error:
                raise RuntimeError(f"CUDA context verification failed: {ctx_check_error}")
            
            try:
                # Set input shape if dynamic
                input_name = self.input_names[0]
                input_shape = preprocessed_image.shape
                
                # Set tensor shapes for dynamic inputs
                if hasattr(self.context, 'set_input_shape'):
                    self.context.set_input_shape(input_name, input_shape)
                elif hasattr(self.context, 'set_binding_shape'):
                    self.context.set_binding_shape(0, input_shape)
                
                # IMPORTANT: Set tensor addresses AFTER setting input shape (if dynamic)
                # This ensures TensorRT knows the correct shapes before we set addresses
                # TensorRT 10.x requires tensor addresses to be set before enqueue/execute
                if hasattr(self.context, 'set_tensor_address'):
                    # Validate device pointers
                    for i, name in enumerate(self.input_names):
                        device_ptr = int(self.inputs[i]['device'])
                        if device_ptr == 0:
                            raise RuntimeError(f"Invalid input device memory pointer for {name} (null)")
                        self.context.set_tensor_address(name, device_ptr)
                    
                    for i, name in enumerate(self.output_names):
                        device_ptr = int(self.outputs[i]['device'])
                        if device_ptr == 0:
                            raise RuntimeError(f"Invalid output device memory pointer for {name} (null)")
                        self.context.set_tensor_address(name, device_ptr)
                
                # Copy input to GPU
                # The preprocessed image should already be in the correct shape (1, 3, H, W)
                host_buffer = self.inputs[0]['host']
                
                # Reshape input to match buffer shape if needed
                input_data = preprocessed_image
                if input_data.shape != host_buffer.shape:
                    # If total size matches, reshape; otherwise it's an error
                    if input_data.size == host_buffer.size:
                        input_data = input_data.reshape(host_buffer.shape)
                    else:
                        raise ValueError(f"Input size {input_data.size} doesn't match buffer size {host_buffer.size}. "
                                       f"Input shape: {preprocessed_image.shape}, Buffer shape: {host_buffer.shape}")
                
                # Copy to host buffer (same shape)
                np.copyto(host_buffer, input_data)
                
                # Get stream handle for async execution
                # CUDA Stream object has a handle attribute that's an integer
                stream_handle = None
                stream_to_sync = None  # Store stream object for synchronization
                
                # Ensure stream is created in current context
                if hasattr(self, 'stream') and self.stream is not None:
                    try:
                        # Verify stream is valid in current context
                        stream_to_sync = self.stream
                        if hasattr(self.stream, 'handle'):
                            stream_handle = self.stream.handle
                        else:
                            stream_handle = int(self.stream)
                    except:
                        # Stream might be invalid in this context, create new one
                        stream_to_sync = cuda.Stream()
                        stream_handle = stream_to_sync.handle if hasattr(stream_to_sync, 'handle') else int(stream_to_sync)
                
                # If no stream available, create one in current context
                if stream_to_sync is None:
                    stream_to_sync = cuda.Stream()
                    stream_handle = stream_to_sync.handle if hasattr(stream_to_sync, 'handle') else int(stream_to_sync)
                
                # Copy input to device (async with stream)
                # Ensure context is valid before memory operations
                cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], stream_to_sync)
                
                # Run inference
                if hasattr(self.context, 'execute_async_v3'):
                    # TensorRT 10.x - requires stream_handle (int), not None
                    self.context.execute_async_v3(stream_handle=stream_handle)
                elif hasattr(self.context, 'execute_async_v2'):
                    # TensorRT 8.x/9.x
                    self.context.execute_async_v2(bindings=[int(self.inputs[0]['device']), int(self.outputs[0]['device'])], stream_handle=stream_handle)
                else:
                    # Fallback to synchronous execution
                    self.context.execute(batch_size=1, bindings=[int(self.inputs[0]['device']), int(self.outputs[0]['device'])])
                
                # Copy output from GPU (async with stream)
                cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], stream_to_sync)
                
                # Synchronize stream to ensure all operations are complete
                stream_to_sync.synchronize()
                
                output = self.outputs[0]['host']
                
                # Reshape output if needed - convert TensorRT Dims to tuple
                output_shape_raw = self.context.get_tensor_shape(self.output_names[0])
                output_shape = self._dims_to_tuple(output_shape_raw)
                
                if output.size != np.prod(output_shape):
                    output = output.reshape(output_shape)
                
                return output
            finally:
                # Clean up context if we pushed it (need to pop)
                if context_pushed:
                    try:
                        current_ctx = cuda.Context.get_current()
                        if current_ctx is not None:
                            current_ctx.pop()
                    except Exception as e:
                        # Ignore errors when popping context
                        pass
    
    def decode_tokens(self, token_ids: np.ndarray) -> str:
        """
        Decode token IDs to text using tokenizer.
        
        Args:
            token_ids: Token IDs from model output
            
        Returns:
            Decoded text string
        """
        if self.tokenizer is None:
            # Fallback: try to decode as ASCII if no tokenizer
            # This is a simple fallback and may not work well
            try:
                # Assume token_ids are character codes
                text = ''.join([chr(int(t)) for t in token_ids.flatten() if 32 <= int(t) <= 126])
                return text
            except:
                return ""
        
        # Use tokenizer to decode
        try:
            # Convert numpy array to list
            if isinstance(token_ids, np.ndarray):
                token_ids = token_ids.flatten().tolist()
            
            # TrOCR uses BERT tokenizer with these special tokens:
            # [PAD] = 0, [UNK] = 100, [CLS] = 101, [SEP] = 102, [MASK] = 103
            # Also, TrOCR may use decoder_start_token_id and eos_token_id
            # Remove special tokens but be more careful - only remove at boundaries
            # The tokenizer.decode with skip_special_tokens=True should handle this,
            # but we'll also manually filter common special tokens
            
            # Filter out common special tokens (but keep the sequence structure)
            # Don't filter too aggressively - let the tokenizer handle it
            filtered_ids = []
            for t in token_ids:
                # Only filter obvious padding/special tokens
                # Keep everything else - tokenizer.decode will handle special tokens correctly
                if t not in [0]:  # Only filter PAD (0), let tokenizer handle others
                    filtered_ids.append(t)
            
            # Decode with skip_special_tokens=True (this handles CLS, SEP, etc.)
            if filtered_ids:
                text = self.tokenizer.decode(filtered_ids, skip_special_tokens=True)
            else:
                text = ""
            
            # Clean up the text: remove extra spaces and normalize
            text = ' '.join(text.split())  # Normalize whitespace
            return text.strip()
        except Exception as e:
            print(f"  [TrOCR] Warning: Token decoding failed: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def recognize(self, image: np.ndarray) -> Dict[str, any]:
        """
        Recognize text in a cropped image region.
        
        Args:
            image: BGR cropped image (H, W, 3) or grayscale (H, W)
            
        Returns:
            Dictionary with 'text' and 'conf' keys
        """
        try:
            # Preprocess
            preprocessed = self.preprocess(image)
            
            # Run inference
            output = self.infer(preprocessed)
            
            # Check output shape to determine format
            # TrOCR encoder output: (batch, sequence_length, hidden_dim) e.g., (1, 577, 768)
            # TrOCR decoder output: (batch, sequence_length, vocab_size) or token IDs
            output_shape = output.shape
            
            # Check output format and decode accordingly
            # Full TrOCR model outputs: token IDs (batch, sequence_length) e.g., (1, 10)
            # Encoder-only outputs: hidden states (batch, sequence_length, hidden_dim) e.g., (1, 577, 768)
            # Decoder logits: (batch, sequence_length, vocab_size) e.g., (1, 10, 50265)
            
            output_shape = output.shape
            
            # Check if this is encoder-only output (3D with last dim 768)
            if len(output_shape) == 3 and output_shape[-1] == 768:
                # This is encoder output (hidden states), not decodable tokens
                print(f"  [TrOCR] Warning: Engine appears to contain only encoder (output shape: {output_shape})")
                print(f"  [TrOCR] Full TrOCR requires both encoder and decoder for text generation")
                print(f"  [TrOCR] Encoder-only models cannot decode text directly")
                print(f"  [TrOCR] Use 'tools/convert_trocr_full_to_onnx.py' to create a full model")
                return {
                    'text': '',
                    'conf': 0.0
                }
            
            # Handle different output formats
            # Full TrOCR encoder-decoder model typically outputs:
            # - Token IDs: (batch, sequence_length) with values in [0, vocab_size)
            # - Or logits: (batch, sequence_length, vocab_size) - need argmax
            
            if len(output_shape) == 2:
                # (batch, sequence_length) or (batch, vocab_size)
                if output_shape[-1] > 1000:
                    # This is logits with large vocab (BERT vocab is ~30k)
                    # Shape is (batch, vocab_size) - single token prediction
                    token_ids = np.argmax(output, axis=-1)
                else:
                    # Likely token IDs: (batch, sequence_length)
                    # Remove batch dimension if present
                    if output_shape[0] == 1:
                        token_ids = output[0]  # Remove batch dim: (sequence_length,)
                    else:
                        token_ids = output
            elif len(output_shape) == 3:
                # Could be logits: (batch, sequence_length, vocab_size)
                if output_shape[-1] > 1000:
                    # Logits - take argmax along vocab dimension (last axis)
                    # Result: (batch, sequence_length)
                    token_ids = np.argmax(output, axis=-1)
                    # Remove batch dimension if present
                    if token_ids.shape[0] == 1:
                        token_ids = token_ids[0]  # (sequence_length,)
                else:
                    # Unknown 3D format - try to use as-is
                    if output_shape[0] == 1:
                        token_ids = output[0]  # Remove batch dim
                    else:
                        token_ids = output
            elif len(output_shape) == 1:
                # 1D token IDs: (sequence_length,)
                token_ids = output
            else:
                # Unknown format, try to flatten and use
                token_ids = output.flatten()
            
            # Ensure token_ids is 1D for decoding
            if len(token_ids.shape) > 1:
                token_ids = token_ids.flatten()
            
            # Debug: print token IDs for troubleshooting
            if len(token_ids) > 0 and len(token_ids) < 50:  # Only for short sequences
                print(f"  [TrOCR] Debug: token_ids shape={token_ids.shape}, sample={token_ids[:10].tolist() if len(token_ids) >= 10 else token_ids.tolist()}")
            
            text = self.decode_tokens(token_ids)
            
            # Debug: print decoded text
            if text:
                print(f"  [TrOCR] Debug: decoded text='{text}'")
            
            # Calculate confidence (simplified - use max probability if logits)
            if len(output.shape) > 1 and output.shape[-1] > 1 and output.shape[-1] < 1000:
                # Use softmax and take max probability
                # Avoid overflow by subtracting max before exp
                output_shifted = output - np.max(output, axis=-1, keepdims=True)
                exp_output = np.exp(output_shifted)
                probs = exp_output / np.sum(exp_output, axis=-1, keepdims=True)
                max_probs = np.max(probs, axis=-1)
                confidence = float(np.mean(max_probs))
            else:
                # If already token IDs, use a default confidence
                confidence = 0.8  # Default confidence for token IDs
            
            return {
                'text': text,
                'conf': confidence
            }
        
        except Exception as e:
            print(f"[TrOCR] Error during recognition: {e}")
            import traceback
            traceback.print_exc()
            return {
                'text': '',
                'conf': 0.0
            }

