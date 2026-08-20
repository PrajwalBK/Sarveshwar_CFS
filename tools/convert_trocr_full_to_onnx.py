"""
Convert Full TrOCR Encoder-Decoder Model to ONNX

This script exports the complete TrOCR model (encoder + decoder) to ONNX format
for TensorRT conversion. The full model can generate text directly from images.
"""

import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import argparse
import os

def convert_trocr_full_to_onnx(
    model_dir: str,
    output_path: str = "models/trocr_full.onnx",
    input_height: int = 384,
    input_width: int = 384,
    max_length: int = 64
):
    """
    Convert full TrOCR encoder-decoder model to ONNX.
    
    Note: TrOCR decoder uses autoregressive generation which is complex to export.
    We'll export a version that generates tokens in a single forward pass.
    
    Args:
        model_dir: Directory containing TrOCR model or Hugging Face model name
        output_path: Output ONNX file path
        input_height: Input image height
        input_width: Input image width
        max_length: Maximum sequence length for decoder output
    """
    print(f"Loading TrOCR model from {model_dir}...")
    
    # Load model
    try:
        if os.path.exists(model_dir):
            processor = TrOCRProcessor.from_pretrained(model_dir)
            model = VisionEncoderDecoderModel.from_pretrained(model_dir)
        else:
            # Try as Hugging Face model name
            processor = TrOCRProcessor.from_pretrained(model_dir)
            model = VisionEncoderDecoderModel.from_pretrained(model_dir)
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Trying to download from Hugging Face...")
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
    
    model.eval()
    
    print(f"Model loaded successfully")
    print(f"  Encoder: {type(model.encoder).__name__}")
    print(f"  Decoder: {type(model.decoder).__name__}")
    
    # Get token IDs from tokenizer
    tokenizer = processor.tokenizer
    print(f"\nToken IDs:")
    print(f"  decoder_start_token_id from config: {getattr(model.config, 'decoder_start_token_id', None)}")
    print(f"  bos_token_id from tokenizer: {getattr(tokenizer, 'bos_token_id', None)}")
    print(f"  cls_token_id from tokenizer: {getattr(tokenizer, 'cls_token_id', None)}")
    print(f"  eos_token_id from config: {getattr(model.config, 'eos_token_id', None)}")
    print(f"  eos_token_id from tokenizer: {getattr(tokenizer, 'eos_token_id', None)}")
    
    # Create dummy input
    dummy_image = torch.randn(1, 3, input_height, input_width)
    
    print(f"\nConverting full TrOCR model to ONNX...")
    print(f"  Input shape: (1, 3, {input_height}, {input_width})")
    print(f"  Max output length: {max_length}")
    print(f"  This may take several minutes...")
    
    # Method 1: Try to export the full model using generate() wrapper
    # However, ONNX doesn't support generate() directly, so we need a custom forward
    
    # Create a wrapper that uses the model's generate method
    # Note: ONNX doesn't support loops well, so we'll use a simplified approach
    class TrOCRWrapper(torch.nn.Module):
        def __init__(self, model, processor, max_length=64):
            super().__init__()
            self.model = model
            self.max_length = max_length
            
            # Get token IDs from tokenizer/config
            # TrOCR uses BERT tokenizer - need to get the correct token IDs
            tokenizer = processor.tokenizer
            
            # Get decoder start token ID
            # TrOCR uses BERT tokenizer - decoder_start_token_id is typically the BOS/CLS token
            decoder_start_token_id = None
            
            # Try config first
            if hasattr(model.config, 'decoder_start_token_id'):
                val = model.config.decoder_start_token_id
                if val is not None:
                    decoder_start_token_id = val
            
            # If None, try tokenizer attributes (check for None explicitly)
            if decoder_start_token_id is None:
                if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
                    decoder_start_token_id = tokenizer.bos_token_id
                elif hasattr(tokenizer, 'cls_token_id') and tokenizer.cls_token_id is not None:
                    decoder_start_token_id = tokenizer.cls_token_id
            
            # If still None, try special tokens map
            if decoder_start_token_id is None and hasattr(tokenizer, 'special_tokens_map'):
                try:
                    special_tokens = tokenizer.special_tokens_map
                    if 'bos_token' in special_tokens and special_tokens['bos_token']:
                        bos_token = special_tokens['bos_token']
                        decoder_start_token_id = tokenizer.convert_tokens_to_ids(bos_token)
                    elif 'cls_token' in special_tokens and special_tokens['cls_token']:
                        cls_token = special_tokens['cls_token']
                        decoder_start_token_id = tokenizer.convert_tokens_to_ids(cls_token)
                except:
                    pass
            
            # Last resort: use pad_token_id or default
            if decoder_start_token_id is None:
                if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
                    decoder_start_token_id = tokenizer.pad_token_id
                else:
                    # Default for BERT-based models: 2 (CLS token)
                    decoder_start_token_id = 2
            
            # Final guarantee - must have a value
            if decoder_start_token_id is None:
                decoder_start_token_id = 2
            
            # Get EOS token ID
            eos_token_id = None
            
            # Try config first
            if hasattr(model.config, 'eos_token_id'):
                eos_token_id = model.config.eos_token_id
            
            # If None, try tokenizer
            if eos_token_id is None:
                if hasattr(tokenizer, 'eos_token_id'):
                    eos_token_id = tokenizer.eos_token_id
                elif hasattr(tokenizer, 'sep_token_id'):
                    eos_token_id = tokenizer.sep_token_id
            
            # If still None, try special tokens map
            if eos_token_id is None and hasattr(tokenizer, 'special_tokens_map'):
                special_tokens = tokenizer.special_tokens_map
                if 'eos_token' in special_tokens and special_tokens['eos_token']:
                    eos_token = special_tokens['eos_token']
                    eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)
                elif 'sep_token' in special_tokens and special_tokens['sep_token']:
                    sep_token = special_tokens['sep_token']
                    eos_token_id = tokenizer.convert_tokens_to_ids(sep_token)
            
            # Last resort: default
            if eos_token_id is None:
                # Default for BERT: 102 (SEP token)
                eos_token_id = 102
            
            # Ensure we have valid integers - force to int and provide defaults
            if decoder_start_token_id is None:
                decoder_start_token_id = 2  # Default BERT CLS token
            try:
                self.decoder_start_token_id = int(decoder_start_token_id)
            except (ValueError, TypeError):
                self.decoder_start_token_id = 2
            
            if eos_token_id is None:
                eos_token_id = 102  # Default BERT SEP token
            try:
                self.eos_token_id = int(eos_token_id)
            except (ValueError, TypeError):
                self.eos_token_id = 102
            
            print(f"  Using decoder_start_token_id: {self.decoder_start_token_id}")
            print(f"  Using eos_token_id: {self.eos_token_id}")
            
            # Final safety check - ensure they're not None
            if self.decoder_start_token_id is None:
                self.decoder_start_token_id = 2
            if self.eos_token_id is None:
                self.eos_token_id = 102
        
        def forward(self, pixel_values):
            # Use model.generate() - but ONNX doesn't support this directly
            # So we'll implement a simplified version
            
            # Encoder forward
            encoder_outputs = self.model.encoder(pixel_values=pixel_values)
            encoder_hidden_states = encoder_outputs.last_hidden_state
            
            # Initialize decoder with BOS token
            batch_size = pixel_values.shape[0]
            device = pixel_values.device
            
            # Use the stored token ID (must be a valid integer)
            # Final safety check
            token_id = self.decoder_start_token_id
            if token_id is None:
                token_id = 2
            token_id = int(token_id)
            
            decoder_input_ids = torch.full(
                (batch_size, 1),
                token_id,
                dtype=torch.long,
                device=device
            )
            
            # Unroll the autoregressive loop (ONNX doesn't support dynamic loops)
            # We'll do max_length steps and let the decoder handle EOS
            generated_ids = decoder_input_ids
            
            # Note: ONNX export with loops is complex. We'll try a fixed unroll
            # For better results, consider using ONNX Runtime with the PyTorch model
            # or exporting encoder/decoder separately
            
            for step in range(self.max_length - 1):
                decoder_outputs = self.model.decoder(
                    input_ids=generated_ids,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=None
                )
                
                # Get next token logits
                next_token_logits = decoder_outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Stop if EOS
                if torch.all(next_token == self.eos_token_id):
                    break
                
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            return generated_ids
    
    # Create wrapper
    wrapper = TrOCRWrapper(model, processor, max_length=max_length)
    wrapper.eval()
    
    # Export to ONNX
    try:
        print(f"\nExporting to ONNX (this may take 5-10 minutes)...")
        torch.onnx.export(
            wrapper,
            dummy_image,
            output_path,
            input_names=["pixel_values"],
            output_names=["generated_ids"],
            dynamic_axes={
                "pixel_values": {0: "batch"},
                "generated_ids": {0: "batch", 1: "sequence_length"}
            },
            opset_version=14,  # TrOCR needs opset 14+
            do_constant_folding=True,
            verbose=False
        )
        print(f"✓ Full TrOCR ONNX model saved: {output_path}")
        return output_path
    except Exception as e:
        print(f"✗ Error exporting full model: {e}")
        print(f"\nNote: Full TrOCR export is complex due to autoregressive decoder.")
        print(f"Alternative: Export encoder and decoder separately, then combine in inference.")
        raise


def convert_trocr_separate_to_onnx(
    model_dir: str,
    encoder_output: str = "models/trocr_encoder.onnx",
    decoder_output: str = "models/trocr_decoder.onnx",
    input_height: int = 384,
    input_width: int = 384
):
    """
    Convert TrOCR encoder and decoder separately to ONNX.
    
    This is an alternative approach that exports encoder and decoder as separate
    ONNX models. You'll need to run them sequentially in your application.
    
    Args:
        model_dir: Directory containing TrOCR model
        encoder_output: Output path for encoder ONNX
        decoder_output: Output path for decoder ONNX
        input_height: Input image height
        input_width: Input image width
    """
    print(f"Loading TrOCR model from {model_dir}...")
    
    # Load model
    if os.path.exists(model_dir):
        processor = TrOCRProcessor.from_pretrained(model_dir)
        model = VisionEncoderDecoderModel.from_pretrained(model_dir)
    else:
        processor = TrOCRProcessor.from_pretrained(model_dir)
        model = VisionEncoderDecoderModel.from_pretrained(model_dir)
    
    model.eval()
    
    # Export encoder
    print(f"\nExporting encoder to ONNX...")
    dummy_image = torch.randn(1, 3, input_height, input_width)
    
    torch.onnx.export(
        model.encoder,
        dummy_image,
        encoder_output,
        input_names=["pixel_values"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "pixel_values": {0: "batch", 2: "height", 3: "width"},
            "last_hidden_state": {0: "batch", 1: "sequence_length"}
        },
        opset_version=14,
        do_constant_folding=True
    )
    print(f"✓ Encoder ONNX saved: {encoder_output}")
    
    # Export decoder (more complex - needs encoder outputs as input)
    print(f"\nExporting decoder to ONNX...")
    # Create dummy encoder outputs
    with torch.no_grad():
        encoder_outputs = model.encoder(pixel_values=dummy_image)
        encoder_hidden_states = encoder_outputs.last_hidden_state
    
    # Get decoder start token ID (same logic as in wrapper)
    tokenizer = processor.tokenizer
    decoder_start_token_id = None
    
    if hasattr(model.config, 'decoder_start_token_id'):
        decoder_start_token_id = model.config.decoder_start_token_id
    
    if decoder_start_token_id is None:
        if hasattr(tokenizer, 'bos_token_id'):
            decoder_start_token_id = tokenizer.bos_token_id
        elif hasattr(tokenizer, 'cls_token_id'):
            decoder_start_token_id = tokenizer.cls_token_id
    
    if decoder_start_token_id is None and hasattr(tokenizer, 'special_tokens_map'):
        special_tokens = tokenizer.special_tokens_map
        if 'bos_token' in special_tokens and special_tokens['bos_token']:
            decoder_start_token_id = tokenizer.convert_tokens_to_ids(special_tokens['bos_token'])
        elif 'cls_token' in special_tokens and special_tokens['cls_token']:
            decoder_start_token_id = tokenizer.convert_tokens_to_ids(special_tokens['cls_token'])
    
    if decoder_start_token_id is None:
        if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
            decoder_start_token_id = tokenizer.pad_token_id
        else:
            decoder_start_token_id = 2  # Default
    
    # Ensure we have a valid integer
    if decoder_start_token_id is None:
        decoder_start_token_id = 2
    try:
        decoder_start_token_id = int(decoder_start_token_id)
    except (ValueError, TypeError):
        decoder_start_token_id = 2
    
    print(f"  Using decoder_start_token_id: {decoder_start_token_id}")
    
    # Final safety check
    if decoder_start_token_id is None:
        decoder_start_token_id = 2
    
    # Dummy decoder input
    decoder_input_ids = torch.full((1, 1), int(decoder_start_token_id), dtype=torch.long)
    
    # Create decoder wrapper
    class DecoderWrapper(torch.nn.Module):
        def __init__(self, decoder):
            super().__init__()
            self.decoder = decoder
        
        def forward(self, input_ids, encoder_hidden_states):
            outputs = self.decoder(
                input_ids=input_ids,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=None
            )
            return outputs.logits
    
    decoder_wrapper = DecoderWrapper(model.decoder)
    decoder_wrapper.eval()
    
    torch.onnx.export(
        decoder_wrapper,
        (decoder_input_ids, encoder_hidden_states),
        decoder_output,
        input_names=["input_ids", "encoder_hidden_states"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence_length"},
            "encoder_hidden_states": {0: "batch", 1: "sequence_length"},
            "logits": {0: "batch", 1: "sequence_length"}
        },
        opset_version=14,
        do_constant_folding=True
    )
    print(f"✓ Decoder ONNX saved: {decoder_output}")
    
    return encoder_output, decoder_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Full TrOCR Model to ONNX")
    parser.add_argument("--model-dir", type=str, default="microsoft/trocr-base-printed",
                       help="Directory containing TrOCR model or Hugging Face model name")
    parser.add_argument("--output", type=str, default="models/trocr_full.onnx",
                       help="Output ONNX file path (for full model)")
    parser.add_argument("--encoder-output", type=str, default="models/trocr_encoder.onnx",
                       help="Output ONNX file path for encoder (if using separate export)")
    parser.add_argument("--decoder-output", type=str, default="models/trocr_decoder.onnx",
                       help="Output ONNX file path for decoder (if using separate export)")
    parser.add_argument("--input-height", type=int, default=384,
                       help="Input image height")
    parser.add_argument("--input-width", type=int, default=384,
                       help="Input image width")
    parser.add_argument("--max-length", type=int, default=64,
                       help="Maximum sequence length for decoder output")
    parser.add_argument("--separate", action="store_true",
                       help="Export encoder and decoder separately (alternative approach)")
    
    args = parser.parse_args()
    
    try:
        if args.separate:
            convert_trocr_separate_to_onnx(
                model_dir=args.model_dir,
                encoder_output=args.encoder_output,
                decoder_output=args.decoder_output,
                input_height=args.input_height,
                input_width=args.input_width
            )
        else:
            convert_trocr_full_to_onnx(
                model_dir=args.model_dir,
                output_path=args.output,
                input_height=args.input_height,
                input_width=args.input_width,
                max_length=args.max_length
            )
    except Exception as e:
        print(f"\n✗ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        print(f"\nTrying separate export as fallback...")
        try:
            convert_trocr_separate_to_onnx(
                model_dir=args.model_dir,
                encoder_output=args.encoder_output,
                decoder_output=args.decoder_output,
                input_height=args.input_height,
                input_width=args.input_width
            )
            print(f"\n✓ Successfully exported encoder and decoder separately")
            print(f"  You'll need to run them sequentially in your application")
        except Exception as e2:
            print(f"\n✗ Separate export also failed: {e2}")
            raise

