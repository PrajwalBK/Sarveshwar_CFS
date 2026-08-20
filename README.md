# Trailer Vision Edge - AI EdgeBox Orion

Production-ready, Jetson-optimized edge application for video-based trailer inventory tracking.

## Quick Start

### Prerequisites
- NVIDIA Jetson Orin (JetPack)
- Python 3.8+
- TensorRT engines: `models/trailer_detector.engine`, `models/ocr_crnn.engine`

### Installation

```bash
sudo apt-get update && sudo apt-get install -y python3-pip python3-opencv git
pip3 install -r requirements.txt
sudo nvpmodel -m 0 && sudo jetson_clocks
```

### Configuration

1. Edit `config/cameras.yaml` with your camera RTSP URLs
2. Calibrate cameras: `python3 tools/calibrate_h.py --image frame.jpg --save config/calib/<camera-id>_h.json`
3. Define parking spots in `config/spots.geojson`
4. Set environment variables (see `.env.example`)

### Running

```bash
python3 -m app.main_trt_demo
```

Or with Docker:
```bash
docker compose build
docker compose up
```

### Dashboard

Access web UI at: `http://<device-ip>:8080/`

## GPU Acceleration

To run the application models (YOLO detector, EasyOCR, oLmOCR, Depth Estimation) completely on the GPU:

1. **Verify GPU/CUDA Status**:
   Run the diagnostic tool to check if PyTorch and CUDA are properly configured in your environment:
   ```bash
   python tools/diagnose_gpu.py
   ```

2. **Configure Device Settings in `.env`**:
   Ensure your `.env` file includes the device configurations:
   ```env
   # General Device Selection ('cuda' or 'cpu')
   DEVICE=cuda
   DETECTOR_DEVICE=cuda
   OCR_DEVICE=cuda
   DEPTH_DEVICE=cuda
   OCR_USE_GPU=1
   ```

3. **Install PyTorch with CUDA support** (if diagnostics report GPU is unavailable):
   - **Windows**: Activate your virtual environment and run:
     ```bash
     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
     ```
     *(Substitute `cu121` with `cu118` or `cu124` depending on your CUDA driver version)*
   - **NVIDIA Jetson**: Do **NOT** install from standard PyPI. Download NVIDIA's official Jetson PyTorch wheels matching your JetPack:
     [NVIDIA Jetson PyTorch Wheels](https://developer.download.nvidia.com/compute/redist/jp/)
   - **Linux Desktop**: Run:
     ```bash
     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
     ```

## Architecture

- **Camera Ingestion**: RTSP/USB via OpenCV/GStreamer
- **Detection**: YOLO (TensorRT) for vehicle/trailer detection
- **Tracking**: ByteTrack for multi-object tracking
- **OCR**: CRNN/PP-OCR (TensorRT) for trailer ID recognition
- **Geometry**: Homography projection (image → world coordinates)
- **Spot Resolution**: GeoJSON polygon matching
- **Outputs**: CSV logs, screenshots, metrics, event bus publishing
- **Integrations**: Azure Service Bus, Kafka, MQTT, S3, Azure Blob, TimescaleDB

## Project Structure

- `app/` - Core application code
- `config/` - Camera configs, homography, parking spots
- `models/` - TensorRT engines
- `tools/` - Calibration utilities
- `web/` - Web dashboard
- `services/ingest/` - REST API for TimescaleDB ingestion
- `observability/` - Prometheus & Grafana configs

See the full runbook for detailed documentation.


