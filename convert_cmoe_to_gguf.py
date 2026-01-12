#!/usr/bin/env python3
"""
Helper script to convert CMoE models to GGUF format.
This script handles the special case of CMoE models which have MoE layers
instead of standard MLP layers.
"""

import os
import sys
import argparse
import json
from pathlib import Path

def check_cmoe_model(model_dir: str) -> dict:
    """Check if the model is a CMoE model and return its configuration."""
    config_path = os.path.join(model_dir, 'config.json')
    if not os.path.exists(config_path):
        return None
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Check if it's a CMoE model
    if 'moe_config' in config:
        return config.get('moe_config', {})
    
    # Check architecture name
    architectures = config.get('architectures', [])
    if 'LlamaCMoEForCausalLM' in architectures:
        return {'moe_type': 'CMoE'}
    
    return None

def main():
    parser = argparse.ArgumentParser(
        description="Convert CMoE model to GGUF format"
    )
    parser.add_argument(
        'model_dir',
        type=str,
        help='Path to the CMoE model directory'
    )
    parser.add_argument(
        '--outfile',
        type=str,
        default=None,
        help='Output GGUF file path (default: model_dir/cmoe_model.gguf)'
    )
    parser.add_argument(
        '--outtype',
        type=str,
        default='f16',
        choices=['f32', 'f16', 'q8_0', 'q4_0', 'q4_1', 'q5_0', 'q5_1', 'q2_k', 'q3_k_s', 'q3_k_m', 'q3_k_l', 'q4_k_s', 'q4_k_m', 'q5_k_s', 'q5_k_m', 'q6_k', 'q8_0'],
        help='Output quantization type'
    )
    
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"Error: Model directory {model_dir} does not exist")
        sys.exit(1)
    
    # Check if it's a CMoE model
    moe_config = check_cmoe_model(str(model_dir))
    if moe_config:
        print(f"Detected CMoE model:")
        print(f"  MoE Type: {moe_config.get('moe_type', 'CMoE')}")
        print(f"  Number of Experts: {moe_config.get('num_experts', 'N/A')}")
        print(f"  Activated Experts: {moe_config.get('num_activated_experts', 'N/A')}")
        print(f"  Shared Experts: {moe_config.get('num_shared_experts', 'N/A')}")
    else:
        print("Warning: This does not appear to be a CMoE model, but proceeding anyway...")
    
    # Determine output file
    if args.outfile:
        outfile = Path(args.outfile)
    else:
        outfile = model_dir / f"cmoe_{args.outtype}.gguf"
    
    # Import and use convert_hf_to_gguf
    try:
        # Add current directory to path to import convert_hf_to_gguf
        sys.path.insert(0, str(Path(__file__).parent))
        from convert_hf_to_gguf import main as convert_main
        
        # Prepare arguments for convert_hf_to_gguf
        # We need to modify sys.argv to pass arguments to convert_hf_to_gguf
        original_argv = sys.argv.copy()
        sys.argv = [
            'convert_hf_to_gguf.py',
            str(model_dir),
            '--outfile', str(outfile),
            '--outtype', args.outtype
        ]
        
        print(f"\nConverting CMoE model to GGUF format...")
        print(f"  Input: {model_dir}")
        print(f"  Output: {outfile}")
        print(f"  Type: {args.outtype}\n")
        
        try:
            convert_main()
            print(f"\n✓ Successfully converted model to {outfile}")
        except Exception as e:
            print(f"\n✗ Error during conversion: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        finally:
            sys.argv = original_argv
            
    except ImportError as e:
        print(f"Error: Could not import convert_hf_to_gguf: {e}")
        print("Make sure convert_hf_to_gguf.py is in the same directory")
        sys.exit(1)

if __name__ == '__main__':
    main()





