# tools/convert_trocr_to_onnx.py
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import argparse

def convert_trocr_to_onnx(
    model_dir: str,
    output_path: str = "models/trocr.onnx",
    input_height: int = 384,
    input_width: int = 384
):
    """
    Convert TrOCR model to ONNX.
    
    Args:
        model_dir: Directory containing TrOCR model
        output_path: Output ONNX file path
        input_height: Input image height
        input_width: Input image width
    """
    print(f"Loading TrOCR model from {model_dir}...")
    
    # Load model
    processor = TrOCRProcessor.from_pretrained(model_dir)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir)
    model.eval()
    
    # Create dummy input (pixel_values from processor)
    dummy_image = torch.randn(1, 3, input_height, input_width)
    
    print(f"Converting to ONNX...")
    print(f"  Input shape: (1, 3, {input_height}, {input_width})")
    
    # Export to ONNX
    torch.onnx.export(
        model.encoder,  # Export encoder only (decoder is more complex)
        dummy_image,
        output_path,
        input_names=["pixel_values"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "pixel_values": {0: "batch", 2: "height", 3: "width"},
            "last_hidden_state": {0: "batch", 1: "sequence_length"}
        },
        opset_version=14,  # TrOCR needs opset 14+
        do_constant_folding=True
    )
    
    print(f"✓ ONNX model saved: {output_path}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert TrOCR to ONNX")
    parser.add_argument("--model-dir", type=str, required=True,
                       help="Directory containing TrOCR model")
    parser.add_argument("--output", type=str, default="models/trocr.onnx",
                       help="Output ONNX file path")
    parser.add_argument("--input-height", type=int, default=384,
                       help="Input image height")
    parser.add_argument("--input-width", type=int, default=384,
                       help="Input image width")
    
    args = parser.parse_args()
    
    convert_trocr_to_onnx(
        model_dir=args.model_dir,
        output_path=args.output,
        input_height=args.input_height,
        input_width=args.input_width
    )