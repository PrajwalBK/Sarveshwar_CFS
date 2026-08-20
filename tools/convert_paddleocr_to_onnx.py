# convert_paddleocr_to_onnx.py
import paddle
from paddle2onnx import convert
import argparse

def convert_paddleocr_to_onnx(
    model_dir: str,
    model_filename: str = "inference.pdmodel",
    params_filename: str = "inference.pdiparams",
    output_path: str = "paddleocr_rec.onnx",
    input_shape: tuple = (1, 3, 32, 320)  # Adjust based on your model
):
    """
    Convert PaddleOCR inference model to ONNX.
    
    Args:
        model_dir: Directory containing PaddleOCR inference model
        model_filename: Model file name (default: inference.pdmodel)
        params_filename: Params file name (default: inference.pdiparams)
        output_path: Output ONNX file path
        input_shape: Input shape (batch, channels, height, width)
    """
    model_path = f"{model_dir}/{model_filename}"
    params_path = f"{model_dir}/{params_filename}"
    
    print(f"Converting PaddleOCR model to ONNX...")
    print(f"  Model: {model_path}")
    print(f"  Params: {params_path}")
    print(f"  Output: {output_path}")
    print(f"  Input shape: {input_shape}")
    
    # Convert to ONNX
    convert(
        model_dir=model_dir,
        model_filename=model_filename,
        params_filename=params_filename,
        save_file=output_path,
        opset_version=11,
        enable_onnx_checker=True,
        input_shape_dict={'x': input_shape}  # Adjust 'x' to your input name
    )
    
    print(f"✓ ONNX model saved: {output_path}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PaddleOCR to ONNX")
    parser.add_argument("--model-dir", type=str, required=True,
                       help="Directory containing PaddleOCR inference model")
    parser.add_argument("--output", type=str, default="paddleocr_rec.onnx",
                       help="Output ONNX file path")
    parser.add_argument("--input-shape", type=str, default="1,3,32,320",
                       help="Input shape as comma-separated values (default: 1,3,32,320)")
    
    args = parser.parse_args()
    
    # Parse input shape
    input_shape = tuple(map(int, args.input_shape.split(',')))
    
    convert_paddleocr_to_onnx(
        model_dir=args.model_dir,
        output_path=args.output,
        input_shape=input_shape
    )