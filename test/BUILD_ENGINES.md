# How to Rebuild TensorRT Engines on Jetson Orin NX

This guide explains how to rebuild TensorRT engines on your Jetson Orin NX device to ensure optimal performance and compatibility.

## Prerequisites

1. **ONNX Models**: You need the ONNX format models:
   - **Legacy Models** (optional):
   - `trailer_detector.onnx` (YOLO detection model)
   - `ocr_crnn.onnx` (OCR recognition model)
   - **PaddleOCR Models** (recommended):
     - `paddleocr_det.onnx` (PaddleOCR text detection model)
     - `paddleocr_rec.onnx` (PaddleOCR text recognition model)

2. **TensorRT Tools**: Already installed with JetPack:
   - `trtexec` (command-line tool)
   - `tensorrt` Python package (via pip)

## Method 1: Using trtexec (Recommended - Fastest)

`trtexec` is NVIDIA's command-line tool for building TensorRT engines. It's the fastest and most reliable method.

### For Detector Engine (YOLO)

```bash
# Navigate to your project directory
cd ~/EdgeOrion

# Build detector engine
trtexec \
  --onnx=models/trailer_detector.onnx \
  --saveEngine=models/trailer_detector.engine \
  --fp16 \
  --workspace=4096 \
  --minShapes=images:1x3x640x640 \
  --optShapes=images:1x3x640x640 \
  --maxShapes=images:1x3x640x640 \
  --verbose
```

### For OCR Engine (CRNN - Legacy)

```bash
# Build OCR engine
trtexec \
  --onnx=models/ocr_crnn.onnx \
  --saveEngine=models/ocr_crnn.engine \
  --fp16 \
  --minShapes=x:1x3x48x1 \
  --optShapes=x:1x3x48x320 \
  --maxShapes=x:1x3x48x320 \
  --verbose
```

**Note**: TensorRT 10.x does not support `--workspace` option. Remove it if you get an error.

### For PaddleOCR Text Detection Engine

```bash
# Build PaddleOCR text detection engine
trtexec \
  --onnx=models/paddleocr_det.onnx \
  --saveEngine=models/paddleocr_det.engine \
  --fp16 \
  --minShapes=x:1x3x640x640 \
  --optShapes=x:1x3x640x640 \
  --maxShapes=x:1x3x640x640 \
  --verbose
```

### For PaddleOCR Text Recognition Engine

```bash
# Build PaddleOCR text recognition engine
trtexec \
  --onnx=models/paddleocr_rec.onnx \
  --saveEngine=models/paddleocr_rec.engine \
  --fp16 \
  --minShapes=x:1x3x48x1 \
  --optShapes=x:1x3x48x320 \
  --maxShapes=x:1x3x48x320 \
  --verbose
```

### trtexec Options Explained

- `--onnx`: Path to input ONNX model
- `--saveEngine`: Output path for TensorRT engine
- `--fp16`: Use FP16 precision (faster, less memory, slight accuracy loss)
- `--fp32`: Use FP32 precision (slower, more memory, full accuracy)
- `--workspace`: GPU memory workspace in MB (4096 = 4GB) - **Note**: Not supported in TensorRT 10.x, remove if you get an error
- `--minShapes/optShapes/maxShapes`: For dynamic shapes (OCR has dynamic width)
- `--verbose`: Show detailed build information

## Method 2: Using Python Script

Use the provided `build_engines.py` script for more control and automation.

```bash
# Build PaddleOCR engines (recommended)
python3 build_engines.py \
  --paddleocr-det-onnx models/paddleocr_det.onnx \
  --paddleocr-rec-onnx models/paddleocr_rec.onnx \
  --output-dir models/ \
  --fp16

# Build legacy engines (YOLO + CRNN)
python3 build_engines.py \
  --detector-onnx models/trailer_detector.onnx \
  --ocr-onnx models/ocr_crnn.onnx \
  --output-dir models/ \
  --fp16

# Build individually
python3 build_engines.py \
  --paddleocr-det-onnx models/paddleocr_det.onnx \
  --output-dir models/ \
  --fp16

python3 build_engines.py \
  --paddleocr-rec-onnx models/paddleocr_rec.onnx \
  --output-dir models/ \
  --fp16
```

## Method 3: Manual Python Conversion

If you need custom conversion logic, you can use TensorRT Python API directly:

```python
import tensorrt as trt

# Create logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Create builder
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, TRT_LOGGER)

# Parse ONNX model
with open("models/trailer_detector.onnx", "rb") as model:
    parser.parse(model.read())

# Configure builder
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB
config.set_flag(trt.BuilderFlag.FP16)  # Enable FP16

# Build engine
engine = builder.build_engine(network, config)

# Save engine
with open("models/trailer_detector.engine", "wb") as f:
    f.write(engine.serialize())
```

## Converting Models to ONNX

If you have PyTorch models, convert them to ONNX first:

### PyTorch to ONNX (YOLO)

```python
import torch
import torch.onnx

# Load your PyTorch model
model = torch.load("trailer_detector.pth")
model.eval()

# Create dummy input
dummy_input = torch.randn(1, 3, 640, 640)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "trailer_detector.onnx",
    input_names=["images"],
    output_names=["output0"],
    dynamic_axes={"images": {0: "batch"}, "output0": {0: "batch"}},
    opset_version=11
)
```

### PyTorch to ONNX (OCR)

```python
import torch
import torch.onnx

# Load OCR model
model = torch.load("ocr_crnn.pth")
model.eval()

# Create dummy input (dynamic width)
dummy_input = torch.randn(1, 3, 48, 320)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "ocr_crnn.onnx",
    input_names=["x"],
    output_names=["fetch_name_0"],
    dynamic_axes={
        "x": {3: "width"},  # Dynamic width dimension
        "fetch_name_0": {1: "sequence_length"}  # Dynamic sequence length
    },
    opset_version=11
)
```

### PaddleOCR to ONNX

If you have PaddleOCR inference models (`.pdmodel` and `.pdiparams`), use the provided conversion script:

```bash
# Convert PaddleOCR detection model
python3 tools/convert_paddleocr_to_onnx.py \
  --model-dir path/to/paddleocr_det_inference \
  --output models/paddleocr_det.onnx \
  --input-shape 1,3,640,640

# Convert PaddleOCR recognition model
python3 tools/convert_paddleocr_to_onnx.py \
  --model-dir path/to/paddleocr_rec_inference \
  --output models/paddleocr_rec.onnx \
  --input-shape 1,3,48,320
```

**Note**: PaddleOCR inference models can be downloaded from:
- [PaddleOCR Model Zoo](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/doc/doc_en/models_list_en.md)
- Look for "Inference Model" downloads (not training checkpoints)

## Verification

After building, verify the engines work:

```bash
# Test all available engines
python3 test_engine.py
```

The script will test:
- YOLO detector engine (if present)
- PaddleOCR text detection engine (if present)
- OCR (CRNN) engine (if present)
- PaddleOCR text recognition engine (if present)

All available engines should show `✓ PASS`.

## Performance Tips

1. **Use FP16**: 2x faster, 2x less memory, minimal accuracy loss
2. **Optimize Workspace**: Larger workspace = better optimization, but uses more memory
3. **Dynamic Shapes**: For OCR, set proper min/opt/max shapes for dynamic width
4. **Build on Target**: Always build engines on the same device you'll run them on

## Troubleshooting

### Out of Memory During Build

```bash
# Reduce workspace size
--workspace=2048  # Instead of 4096
```

### Dynamic Shape Issues

```bash
# Ensure min/opt/max shapes are set correctly
# Check your model's expected input shapes
```

### Build Fails

1. Check ONNX model is valid: `onnx.checker.check_model("model.onnx")`
2. Verify TensorRT version: `python3 -c "import tensorrt; print(tensorrt.__version__)"`
3. Check GPU memory: `nvidia-smi`
4. Try FP32 instead of FP16: `--fp32`

## Expected Build Times

- **Detector Engine**: 5-15 minutes (depending on model complexity)
- **OCR Engine**: 2-5 minutes

## Next Steps

After rebuilding engines:
1. Test with `python3 test_engine.py`
2. Run application: `python3 -m app.main_trt_demo`
3. Monitor performance and verify no warnings about device mismatch



