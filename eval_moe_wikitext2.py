import argparse
import os
import sys

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from datautils import get_loaders
from run_cmoe import cmoe_ppl_eval, DEV


def load_moe_model(moe_dir: str, device: torch.device):
    """
    Load the carved MoE model using standard HuggingFace AutoModelForCausalLM.
    
    The model directory should contain:
    - config.json (with auto_map pointing to modeling_cmoe.LlamaCMoEForCausalLM)
    - modeling_cmoe.py (custom model architecture)
    - CMoE_model.py, CMoE_utils.py (dependencies)
    - model.safetensors or pytorch_model.bin (model weights)
    - moe_model.pt (fallback: full model object)
    """
    # Add model directory to Python path so modeling_cmoe.py can import dependencies
    if moe_dir not in sys.path:
        sys.path.insert(0, moe_dir)
    
    # Check if modeling_cmoe.py exists (standard HuggingFace way)
    modeling_file = os.path.join(moe_dir, 'modeling_cmoe.py')
    if os.path.exists(modeling_file):
        print(f"Loading CMoE model using custom architecture from {modeling_file}")
        try:
            # Use AutoModelForCausalLM which will automatically use the custom class
            # specified in config.json's auto_map
            device_map = 'auto' if device.type == 'cuda' else None
            model = AutoModelForCausalLM.from_pretrained(
                moe_dir,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                device_map=device_map
            )
            if device_map is None:
                model = model.to(device)
            model.eval()
            print("Successfully loaded model using AutoModelForCausalLM")
            return model
        except Exception as e:
            print(f"Failed to load using AutoModelForCausalLM: {e}")
            print("Falling back to moe_model.pt...")
    
    # Fallback: load from .pt file (for backward compatibility)
    model_path = os.path.join(moe_dir, "moe_model.pt")
    if os.path.exists(model_path):
        print(f"Loading model from {model_path} (fallback method)")
        model = torch.load(model_path, map_location=device)
        model.to(device)
        model.eval()
        return model
    else:
        raise FileNotFoundError(
            f"Could not find modeling_cmoe.py or moe_model.pt in {moe_dir}. "
            f"Please re-run `run_cmoe.py` with --output-dir to regenerate the saved MoE model."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a saved CMoE model on WikiText-2 and report perplexity."
    )
    parser.add_argument(
        "moe_dir",
        type=str,
        help="Path to the saved MoE model directory (created via --output-dir).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run evaluation on (e.g., 'cuda:0' or 'cpu').",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for data loader.",
    )

    args = parser.parse_args()

    device = torch.device(args.device)

    # 1. Load tokenizer and config to get base model path
    tokenizer = AutoTokenizer.from_pretrained(args.moe_dir, use_fast=False)

    # Read moe_config from config.json to recover base model path
    import json

    config_path = os.path.join(args.moe_dir, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)

    moe_cfg = config_dict.get("moe_config", {})
    base_model_path = moe_cfg.get("base_model", None)
    if base_model_path is None:
        raise ValueError(
            "config.json in the MoE directory does not contain 'moe_config.base_model'. "
            "Please re-run `run_cmoe.py` with the updated `save_moe_model` to regenerate the model."
        )

    # 2. Load the carved MoE model (.pt)
    model = load_moe_model(args.moe_dir, device)

    # 3. Prepare WikiText-2 evaluation data
    #    We follow the same pattern as in `run_cmoe.py` / `cmoe_ppl_eval`.
    from datautils import get_loaders

    dataset = "wikitext2"
    _, testloader = get_loaders(
        dataset,
        seed=args.seed,
        model=base_model_path,
        seqlen=model.seqlen,
    )

    # 4. Run perplexity evaluation
    ppl = cmoe_ppl_eval(model, testloader, device, dataset, args=None)
    print(f"WikiText-2 PPL of MoE model in {args.moe_dir}: {ppl}")


if __name__ == "__main__":
    main()


