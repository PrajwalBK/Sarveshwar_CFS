"""
Export Trained OCR CRNN Model to ONNX

Exports a trained CRNN model to ONNX format for TensorRT conversion.
"""

import torch
import torch.nn as nn
import os
from pathlib import Path

# CRNN Model (same as in training script)
class CRNN(nn.Module):
    def __init__(self, img_height, num_channels, num_classes, hidden_size=256):
        super(CRNN, self).__init__()
        
        # CNN backbone
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU(),
        )
        
        # RNN (BiLSTM)
        self.rnn = nn.LSTM(512, hidden_size, bidirectional=True, num_layers=2, batch_first=True)
        
        # Classifier
        self.classifier = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, x):
        # CNN
        conv = self.cnn(x)  # [B, 512, 1, W']
        b, c, h, w = conv.size()
        conv = conv.squeeze(2).permute(0, 2, 1)  # [B, W', 512]
        
        # RNN
        rnn_out, _ = self.rnn(conv)  # [B, W', hidden*2]
        
        # Classifier
        output = self.classifier(rnn_out)  # [B, W', num_classes]
        output = output.permute(1, 0, 2)  # [W', B, num_classes] for CTC
        
        return output


def load_alphabet(path):
    """Load alphabet from file."""
    with open(path, 'r') as f:
        alphabet = f.read().strip()
    return alphabet


def export_to_onnx(
    weights_path: str = "models/ocr_crnn_final.pth",
    alphabet_path: str = "training_data/alphabet.txt",
    output_path: str = "models/ocr_crnn.onnx",
    img_height: int = 32,
    img_width: int = 320
):
    """
    Export trained CRNN model to ONNX.
    
    Args:
        weights_path: Path to trained .pth weights file
        alphabet_path: Path to alphabet.txt file
        output_path: Output path for ONNX file
        img_height: Image height (default: 32)
        img_width: Image width (default: 320)
    """
    print("=" * 60)
    print("OCR CRNN Model Export to ONNX")
    print("=" * 60)
    
    # Check if weights file exists
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    
    # Load alphabet to determine num_classes
    alphabet = load_alphabet(alphabet_path)
    num_classes = len(alphabet) + 1  # +1 for blank
    
    print(f"\n[1/4] Loaded alphabet: {alphabet} ({len(alphabet)} chars)")
    print(f"  Num classes: {num_classes}")
    
    # Create model
    print(f"\n[2/4] Creating model...")
    model = CRNN(img_height, 1, num_classes)  # 1 channel (grayscale)
    
    # Load weights
    print(f"\n[3/4] Loading weights from: {weights_path}")
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    print(f"  ✓ Model loaded successfully")
    
    # Create output directory if needed
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dummy input
    dummy_input = torch.randn(1, 1, img_height, img_width)
    
    print(f"\n[4/4] Exporting to ONNX...")
    print(f"  Output: {output_path}")
    print(f"  Input shape: (1, 1, {img_height}, {img_width})")
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["x"],
        output_names=["output"],
        dynamic_axes={
            "x": {3: "width"},  # Dynamic width dimension
            "output": {0: "sequence"}  # Dynamic sequence length
        },
        opset_version=11,
        do_constant_folding=True,
        verbose=False
    )
    
    print(f"  ✓ ONNX export complete!")
    
    # Verify file exists
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"  ✓ File size: {file_size:.2f} MB")
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Export trained OCR CRNN model to ONNX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export with default paths
  python tools/export_ocr_to_onnx.py
  
  # Export with custom paths
  python tools/export_ocr_to_onnx.py --weights models/ocr_crnn_epoch_100.pth --output models/my_ocr.onnx
        """
    )
    
    parser.add_argument(
        "--weights",
        type=str,
        default="models/ocr_crnn_final.pth",
        help="Path to trained .pth weights file (default: models/ocr_crnn_final.pth)"
    )
    
    parser.add_argument(
        "--alphabet",
        type=str,
        default="training_data/alphabet.txt",
        help="Path to alphabet.txt file (default: training_data/alphabet.txt)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="models/ocr_crnn.onnx",
        help="Output path for ONNX file (default: models/ocr_crnn.onnx)"
    )
    
    parser.add_argument(
        "--img-height",
        type=int,
        default=32,
        help="Image height (default: 32)"
    )
    
    parser.add_argument(
        "--img-width",
        type=int,
        default=320,
        help="Image width (default: 320)"
    )
    
    args = parser.parse_args()
    
    try:
        export_to_onnx(
            weights_path=args.weights,
            alphabet_path=args.alphabet,
            output_path=args.output,
            img_height=args.img_height,
            img_width=args.img_width
        )
        
        print("\n" + "=" * 60)
        print("✓ Export completed successfully!")
        print("=" * 60)
        print(f"\nNext steps:")
        print(f"1. Build TensorRT engine on Jetson:")
        print(f"   trtexec --onnx={args.output} --saveEngine=models/ocr_crnn.engine --fp16")
        print(f"2. Or use build_engines.py:")
        print(f"   python build_engines.py --ocr-onnx {args.output} --fp16")
        
    except Exception as e:
        print(f"\n[Export] Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)