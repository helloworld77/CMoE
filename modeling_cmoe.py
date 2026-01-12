"""
Custom model architecture for CMoE (Carved Mixture-of-Experts) models.
This file should be saved alongside the model weights to enable proper loading.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from transformers import LlamaForCausalLM, LlamaModel, LlamaPreTrainedModel
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaConfig

# Import CMoE components
# Handle both cases: when imported from project root and when imported from model directory
try:
    from CMoE_model import MoE, Router, LlamaMLP
except ImportError:
    # If imported from model directory, try importing from current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    from CMoE_model import MoE, Router, LlamaMLP


class LlamaCMoEDecoderLayer(LlamaDecoderLayer):
    """
    LlamaDecoderLayer with MoE replacing the MLP.
    The MLP is replaced with a MoE module during model carving.
    """
    
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        # MLP will be replaced with MoE during model construction
        # We keep the original structure but MoE will be injected


class LlamaCMoEModel(LlamaModel):
    """
    LlamaModel with CMoE decoder layers.
    """
    
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        # Replace decoder layers with CMoE versions
        self.layers = nn.ModuleList([
            LlamaCMoEDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])


class LlamaCMoEForCausalLM(LlamaForCausalLM):
    """
    LlamaForCausalLM with CMoE architecture.
    This is the main model class that should be used for loading CMoE models.
    """
    
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        # Replace the model with CMoE version
        self.model = LlamaCMoEModel(config)
        # Initialize weights
        self.post_init()
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *model_args,
        **kwargs
    ):
        """
        Load a CMoE model from pretrained path.
        
        Priority:
        1. Load from moe_model.pt (full model object, preserves exact structure)
        2. Load from safetensors/pytorch weights and reconstruct architecture
        """
        model_dir = pretrained_model_name_or_path
        if not os.path.isdir(model_dir):
            model_dir = pretrained_model_name_or_path
        
        # First, try to load from .pt file (preserves exact MoE structure)
        pt_path = os.path.join(model_dir, 'moe_model.pt')
        if os.path.exists(pt_path):
            print(f"Loading CMoE model from {pt_path}")
            model = torch.load(pt_path, map_location='cpu')
            # Apply device_map or move to device
            device_map = kwargs.get('device_map', None)
            if device_map:
                # Let transformers handle device_map
                pass
            else:
                # Move to specified device or default
                device = kwargs.get('device', 'cpu')
                if isinstance(device, str):
                    device = torch.device(device)
                model = model.to(device)
            return model
        
        # Fallback: Load config and reconstruct model, then load weights
        from transformers import AutoConfig
        from safetensors.torch import load_file as safe_load_file
        import json
        
        # Load config
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        
        # Create model instance (note: this creates a fresh model with MLP, not MoE)
        # We'll load the weights which contain MoE structure
        model = cls(config)
        
        # Load state dict from safetensors or pytorch format
        if os.path.isdir(model_dir):
            state_dict = {}
            
            # Try sharded safetensors first
            index_path = os.path.join(model_dir, 'model.safetensors.index.json')
            if os.path.exists(index_path):
                with open(index_path, 'r') as f:
                    index = json.load(f)
                weight_map = index.get('weight_map', {})
                # Get unique shard files
                shard_files = sorted(set(weight_map.values()))
                for shard_file in shard_files:
                    shard_path = os.path.join(model_dir, shard_file)
                    if os.path.exists(shard_path):
                        shard_dict = safe_load_file(shard_path)
                        state_dict.update(shard_dict)
            # Try single safetensors file
            elif os.path.exists(os.path.join(model_dir, 'model.safetensors')):
                safetensors_path = os.path.join(model_dir, 'model.safetensors')
                state_dict = safe_load_file(safetensors_path)
            # Try pytorch format
            elif os.path.exists(os.path.join(model_dir, 'pytorch_model.bin')):
                pytorch_path = os.path.join(model_dir, 'pytorch_model.bin')
                state_dict = torch.load(pytorch_path, map_location='cpu')
            else:
                raise FileNotFoundError(
                    f"Could not find model weights in {model_dir}. "
                    f"Expected moe_model.pt, model.safetensors, model.safetensors.index.json, or pytorch_model.bin"
                )
            
            # Load state dict into model
            # Note: Since the saved model has MoE structure but we're creating a fresh model,
            # we need to handle this carefully. The weights should match the MoE structure.
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"Warning: Missing keys when loading state dict: {len(missing_keys)} keys")
                if len(missing_keys) <= 20:
                    print(f"  Missing: {missing_keys}")
            if unexpected_keys:
                print(f"Warning: Unexpected keys when loading state dict: {len(unexpected_keys)} keys")
                if len(unexpected_keys) <= 20:
                    print(f"  Unexpected: {unexpected_keys}")
        else:
            # Fallback to standard transformers loading
            from transformers import AutoModelForCausalLM
            base_model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                **{k: v for k, v in kwargs.items() if k != 'torch_dtype'}
            )
            model.load_state_dict(base_model.state_dict(), strict=False)
        
        return model

