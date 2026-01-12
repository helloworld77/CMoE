#!/usr/bin/env python3
"""
Evaluate GGUF format CMoE model on WikiText-2 perplexity.

This script supports two methods:
1. Using llama-cpp-python (recommended for Python integration)
2. Using llama.cpp command-line tool (if llama-cpp-python is not available)
"""

import argparse
import os
import sys
import subprocess
import json
from pathlib import Path


def eval_with_llama_cpp_python(gguf_path: str, testloader, seqlen: int = 2048):
    """
    Evaluate using llama-cpp-python library.
    This method processes the entire sequence at once for efficiency.
    """
    try:
        from llama_cpp import Llama
    except ImportError:
        raise ImportError(
            "llama-cpp-python is not installed. "
            "Install it with: pip install llama-cpp-python"
        )
    
    print(f"Loading GGUF model from {gguf_path}...")
    llm = Llama(
        model_path=gguf_path,
        n_ctx=seqlen,
        verbose=False,
        n_threads=os.cpu_count(),
        logits_all=True,  # Get logits for all tokens
    )
    
    print("Loading test data...")
    testenc = testloader.input_ids
    
    nsamples = testenc.numel() // seqlen
    print(f"Evaluating on {nsamples} samples (seqlen={seqlen})...")
    
    nlls = []
    from tqdm import tqdm
    
    for i in tqdm(range(nsamples), desc="Computing perplexity"):
        batch = testenc[:, (i * seqlen):((i + 1) * seqlen)]
        batch_list = batch.squeeze(0).tolist()
        
        # Get logits for the entire sequence
        result = llm(
            batch_list,
            max_tokens=0,  # Don't generate, just get logits
            logprobs=1,
            temperature=0.0,
        )
        
        # Extract log probabilities
        if 'choices' in result and len(result['choices']) > 0:
            choice = result['choices'][0]
            if 'logprobs' in choice and 'token_logprobs' in choice['logprobs']:
                token_logprobs = choice['logprobs']['token_logprobs']
                # Skip the first token (no context)
                if len(token_logprobs) > 1:
                    log_probs = token_logprobs[1:]  # Skip first token
                    neg_log_likelihood = -sum(log_probs) * seqlen
                    nlls.append(neg_log_likelihood)
    
    if nlls:
        import torch
        ppl = torch.exp(torch.tensor(nlls).sum() / (len(nlls) * seqlen))
        return ppl.item()
    else:
        raise ValueError("Failed to compute log probabilities")


def eval_with_llama_cpp_cli(gguf_path: str, testloader, base_model_path: str, llama_cpp_path: str = None, seqlen: int = 2048):
    """
    Evaluate using llama.cpp command-line tool.
    Requires llama.cpp to be built with perplexity tool.
    This is the recommended method as it's more efficient.
    """
    # Try to find llama.cpp binary
    if llama_cpp_path is None:
        possible_paths = [
            "./llama.cpp/build/bin/perplexity",
            "./llama.cpp/perplexity",
            "../llama.cpp/build/bin/perplexity",
            "../llama.cpp/perplexity",
            "~/llama.cpp/build/bin/perplexity",
            "/usr/local/bin/perplexity",
            "perplexity",  # In PATH
        ]
        for path in possible_paths:
            expanded_path = os.path.expanduser(path)
            if os.path.exists(expanded_path) and os.access(expanded_path, os.X_OK):
                llama_cpp_path = expanded_path
                break
        
        if llama_cpp_path is None:
            raise FileNotFoundError(
                "Could not find llama.cpp perplexity binary. "
                "Please specify --llama-cpp-path or install llama-cpp-python.\n"
                "To build llama.cpp, see: https://github.com/ggerganov/llama.cpp"
            )
    
    # Convert test data to text format for llama.cpp
    from transformers import AutoTokenizer
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)
    
    # Decode test data (same as in datautils)
    testenc = testloader.input_ids
    test_text = tokenizer.decode(testenc.squeeze(0), skip_special_tokens=False)
    
    # Save test data as temporary text file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        test_text_path = f.name
        f.write(test_text)
    
    print(f"Test data saved to temporary file: {test_text_path}")
    
    # Run llama.cpp perplexity
    cmd = [
        llama_cpp_path,
        "-m", gguf_path,
        "-f", test_text_path,
        "-c", str(seqlen),  # context length
        "-ngl", "0",  # No GPU layers (use CPU), change if you have GPU support
    ]
    
    print(f"\nRunning: {' '.join(cmd)}")
    print("This may take a while...\n")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Clean up temp file
    if os.path.exists(test_text_path):
        os.remove(test_text_path)
    
    if result.returncode != 0:
        print(f"Error output: {result.stderr}")
        raise RuntimeError(f"llama.cpp perplexity failed with return code {result.returncode}")
    
    # Parse output to extract perplexity
    output = result.stdout
    print("llama.cpp output:")
    print(output)
    print()
    
    # Look for perplexity value in output
    import re
    # Common patterns: "perplexity: 12.34" or "ppl: 12.34" or "12.34 ppl"
    patterns = [
        r'perplexity[:\s]+([\d.]+)',
        r'ppl[:\s]+([\d.]+)',
        r'([\d.]+)\s+ppl',
        r'([\d.]+)\s+perplexity',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, output, re.IGNORECASE)
        if matches:
            try:
                return float(matches[-1])  # Take the last match (usually the final result)
            except ValueError:
                continue
    
    raise ValueError(f"Could not parse perplexity from output. Please check the output above.")


def get_base_model_path(model_dir: str) -> str:
    """
    Get base model path from config.json in model directory.
    """
    config_path = os.path.join(model_dir, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    moe_config = config.get('moe_config', {})
    base_model_path = moe_config.get('base_model', None)
    if base_model_path is None:
        # Fallback: try to use model_dir itself
        base_model_path = model_dir
    
    return base_model_path


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GGUF format CMoE model on WikiText-2 perplexity"
    )
    parser.add_argument(
        'gguf_path',
        type=str,
        help='Path to the GGUF model file (.gguf)'
    )
    parser.add_argument(
        '--method',
        type=str,
        choices=['llama-cpp-python', 'llama-cpp-cli', 'auto'],
        default='auto',
        help='Evaluation method (default: auto, tries llama-cpp-python first)'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default=None,
        help='Path to original model directory (for loading tokenizer and base model path). If not provided, will infer from GGUF path.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed for data loader (default: 0)'
    )
    parser.add_argument(
        '--llama-cpp-path',
        type=str,
        default=None,
        help='Path to llama.cpp perplexity binary (only needed for llama-cpp-cli method)'
    )
    parser.add_argument(
        '--seqlen',
        type=int,
        default=2048,
        help='Sequence length for evaluation (default: 2048)'
    )
    
    args = parser.parse_args()
    
    # Determine model directory from GGUF path if not provided
    if args.model_dir is None:
        args.model_dir = os.path.dirname(os.path.abspath(args.gguf_path))
    
    # Get base model path from config
    print(f"Loading model configuration from {args.model_dir}...")
    base_model_path = get_base_model_path(args.model_dir)
    print(f"Base model path: {base_model_path}")
    
    # Load test data using the same method as in run_cmoe.py
    from datautils import get_loaders
    
    print("Loading WikiText-2 test data...")
    dataset = "wikitext2"
    _, testloader = get_loaders(
        dataset,
        seed=args.seed,
        model=base_model_path,
        seqlen=args.seqlen,
    )
    print(f"Test data loaded: {testloader.input_ids.shape}")
    
    # Determine evaluation method
    if args.method == 'auto':
        try:
            import llama_cpp
            method = 'llama-cpp-python'
            print("Using llama-cpp-python method")
        except ImportError:
            method = 'llama-cpp-cli'
            print("llama-cpp-python not available, trying llama-cpp-cli method")
    else:
        method = args.method
    
    # Run evaluation
    try:
        if method == 'llama-cpp-python':
            ppl = eval_with_llama_cpp_python(args.gguf_path, testloader, args.seqlen)
        elif method == 'llama-cpp-cli':
            ppl = eval_with_llama_cpp_cli(args.gguf_path, testloader, base_model_path, args.llama_cpp_path, args.seqlen)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        print(f"\n{'='*60}")
        print(f"WikiText-2 Perplexity: {ppl:.4f}")
        print(f"{'='*60}\n")
        
        return ppl
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

