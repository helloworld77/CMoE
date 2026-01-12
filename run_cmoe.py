import os 
import time
import copy
import argparse
import json
import shutil
from datautils import *
from tqdm import *
import torch
import torch.nn as nn
from safetensors.torch import save_file as safe_save_file
from collections import OrderedDict
from transformers import AutoTokenizer 

from CMoE_utils import *
from CMoE_model import *
from zero_eval import *
from sft_utils import simple_sft

DEV = torch.device('cuda:0')

def get_llama(model):
    def skip(*args, **kwargs):
        pass
    # torch.nn.init.kaiming_uniform_ = skip
    # torch.nn.init.uniform_ = skip
    # torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map = 'auto')
    model.seqlen = 2048
    return model

def get_llava(model):
    def skip(*args, **kwargs):
        pass
    # torch.nn.init.kaiming_uniform_ = skip
    # torch.nn.init.uniform_ = skip
    # torch.nn.init.normal_ = skip

    from llava.model import LlavaLlamaForCausalLM

    model = LlavaLlamaForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map = 'auto')
    model.seqlen = 2048
    return model


def cmoe_sequential(model, dataloader, dev, args):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    dtype = next(iter(model.parameters())).dtype
    bsz = 8
    
    inps = torch.zeros(
        (args.nsamples//bsz, bsz, model.seqlen, model.config.hidden_size), dtype=dtype, device='cpu'
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):

            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    
    with torch.no_grad():
        for batch in dataloader:
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass

    layers[0] = layers[0].module

    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    moe_outs = torch.zeros_like(inps)
    
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')
    model.cuda()
    layers.cuda()

    inp = copy.deepcopy(inps[0])

    # MoE Carving
    carve_inp = copy.deepcopy(inp)
    for layer in tqdm(layers, desc = 'Carving MoE layers...'):
        moe_out = construct_moe(layer, 
            carve_inp, 
            attention_mask, 
            position_ids,
            n_experts = args.nexperts,
            n_activated = args.nactivated,
            n_shared = args.nshared,
            args = args
        )
        carve_inp = moe_out

    
    tick_1 = time.time()

    print('Training_free_ppl:')
    pre_ppl = []
    datasets = ['wikitext2'] #, 'c4-new']
    for dataset in datasets:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print(dataset)
        eval_set = dataset
        ppl_i = cmoe_ppl_eval(model, testloader, DEV, eval_set, args)
        pre_ppl.append(f"{dataset}: {ppl_i}")
    
    tick_2 = time.time()
    

    # LoRa-based Supervised Fine-tuning
    for layer in layers:
        layer.mlp.cus_training = True

    model.cuda()
    model = simple_sft(model, args, epoch = args.epoch)

    for layer in layers:
        layer.mlp.cus_training = False

    model.eval()

    model.config.use_cache = use_cache
    
    return model, tick_1, tick_2, pre_ppl

@torch.no_grad()
def cmoe_ppl_eval(model, testenc, dev, eval_set, args):
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()
    
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    for i in tqdm(range(len(layers)), desc= 'Processing...'):

        layer = layers[i].to(dev)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())
    model.config.use_cache = use_cache

    return ppl.item()

def save_results(file_name, results):
    if results is not str:
        results = str(results)
    results = results + '\n'
    if not os.path.exists(file_name):
        with open(file_name, "w") as file:
            file.write(results)
    else:
        with open(file_name, "a") as file:
            file.write(results)

def save_moe_model(model, tokenizer, output_dir, args):
    """
    Save the converted MoE model with all necessary files:
    - config.json (with MoE parameters)
    - model.safetensors (model weights)
    - tokenizer files
    """    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Update config with MoE parameters
    config = model.config
    config_dict = config.to_dict()
    
    # Add MoE-specific configuration
    # NOTE: we also store the base dense model path for later evaluation/loading.
    config_dict['moe_config'] = {
        'num_experts': args.nexperts,
        'num_activated_experts': args.nactivated,
        'num_shared_experts': args.nshared,
        'moe_type': 'CMoE',
        'intermediate_size_per_expert': config.intermediate_size // args.nexperts,
        'base_model': args.model,
    }
    
    # Configure auto_map to use custom model class
    # This tells transformers to use our custom LlamaCMoEForCausalLM class
    config_dict['auto_map'] = {
        'AutoModelForCausalLM': 'modeling_cmoe.LlamaCMoEForCausalLM'
    }
    config_dict['architectures'] = ['LlamaCMoEForCausalLM']
    
    # Save updated config
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    # Copy modeling_cmoe.py to model directory so it can be imported during loading
    modeling_file_src = os.path.join(os.path.dirname(__file__), 'modeling_cmoe.py')
    modeling_file_dst = os.path.join(output_dir, 'modeling_cmoe.py')
    if os.path.exists(modeling_file_src):
        shutil.copy2(modeling_file_src, modeling_file_dst)
        print(f"Model architecture file saved to {modeling_file_dst}")
    else:
        print(f"Warning: Could not find {modeling_file_src}, model loading may fail")
    
    # Also copy CMoE_model.py and CMoE_utils.py as dependencies
    for dep_file in ['CMoE_model.py', 'CMoE_utils.py']:
        dep_src = os.path.join(os.path.dirname(__file__), dep_file)
        dep_dst = os.path.join(output_dir, dep_file)
        if os.path.exists(dep_src):
            shutil.copy2(dep_src, dep_dst)
            print(f"Dependency file {dep_file} saved to {dep_dst}")
    
    # Save model weights using safetensors format
    print("Saving model weights...")
    try:
        # Collect all state dict
        state_dict = model.state_dict()
        
        # Convert to safetensors format
        # Split into multiple files if needed (for large models)
        max_shard_size = 5 * 1024 * 1024 * 1024  # 5GB per shard
        
        # Calculate total size
        total_size = sum(p.numel() * p.element_size() for p in model.parameters())
        
        if total_size > max_shard_size:
            # Split into multiple shards
            shard_files = []
            current_shard = OrderedDict()
            current_size = 0
            shard_index = 1
            weight_map = {}
            
            for key, value in state_dict.items():
                param_size = value.numel() * value.element_size()
                
                if current_size + param_size > max_shard_size and len(current_shard) > 0:
                    # Save current shard
                    total_shards = len(state_dict)  # Estimate, will update later
                    shard_file = os.path.join(output_dir, f'model-{shard_index:05d}-of-{total_shards:05d}.safetensors')
                    safe_save_file(current_shard, shard_file)
                    shard_files.append(shard_file)
                    
                    # Update weight_map for all keys in current shard
                    for k in current_shard.keys():
                        weight_map[k] = os.path.basename(shard_file)
                    
                    current_shard = OrderedDict()
                    current_size = 0
                    shard_index += 1
                
                current_shard[key] = value
                current_size += param_size
            
            # Save last shard
            if len(current_shard) > 0:
                total_shards = shard_index
                shard_file = os.path.join(output_dir, f'model-{shard_index:05d}-of-{total_shards:05d}.safetensors')
                safe_save_file(current_shard, shard_file)
                shard_files.append(shard_file)
                
                # Update weight_map for all keys in last shard
                for k in current_shard.keys():
                    weight_map[k] = os.path.basename(shard_file)
            
            # Create model.safetensors.index.json
            index_dict = {
                "metadata": {"total_size": total_size},
                "weight_map": weight_map
            }
            
            index_path = os.path.join(output_dir, 'model.safetensors.index.json')
            with open(index_path, 'w', encoding='utf-8') as f:
                json.dump(index_dict, f, indent=2)
            print(f"Model weights saved in {len(shard_files)} shards")
        else:
            # Single file
            model_file = os.path.join(output_dir, 'model.safetensors')
            safe_save_file(state_dict, model_file)
            print(f"Model weights saved to {model_file}")
    
    except ImportError:
        print("Warning: safetensors not available, saving in PyTorch format...")
        # Fallback to PyTorch format
        model_file = os.path.join(output_dir, 'pytorch_model.bin')
        torch.save(model.state_dict(), model_file)
        print(f"Model weights saved to {model_file}")

    # Additionally, save the full model object for easy re-loading in local experiments.
    # This .pt file is Python/pickle specific (not ideal for distribution), but very convenient
    # when you just want to quickly evaluate the exact carved MoE model.
    full_model_path = os.path.join(output_dir, 'moe_model.pt')
    torch.save(model, full_model_path)
    print(f"Full MoE model object saved to {full_model_path}")
    
    # Save tokenizer files
    print("Saving tokenizer...")
    tokenizer.save_pretrained(output_dir)
    
    # Save a README with model information
    readme_content = f"""# CMoE Model

This model was converted from a dense LLM to a Sparse MoE architecture using CMoE.

## Model Configuration

- Base Model: {args.model}
- Number of Experts: {args.nexperts}
- Number of Activated Experts: {args.nactivated}
- Number of Shared Experts: {args.nshared}
- Calibration Dataset: {args.dataset}
- Number of Calibration Samples: {args.nsamples}
- Fine-tuning Epochs: {args.epoch}

## Model Structure

The original Dense FFN layers have been replaced with MoE (Mixture of Experts) layers.
Each MoE layer contains:
- {args.nexperts} experts total ({args.nshared} shared + {args.nexperts - args.nshared} routed)
- {args.nactivated} experts activated per token
- Router for expert selection

## Loading the Model

### Standard HuggingFace Way (Recommended)

The model can be loaded using standard `AutoModelForCausalLM.from_pretrained()`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "{output_dir}",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("{output_dir}")

# The model will automatically use the custom LlamaCMoEForCausalLM architecture
# defined in modeling_cmoe.py (specified via auto_map in config.json)
```

### Fallback Method

If the standard loading fails, you can use the pickled model:

```python
import torch
model = torch.load("{output_dir}/moe_model.pt")
```

## Files

- `config.json`: Model configuration with MoE parameters and auto_map
- `modeling_cmoe.py`: Custom model architecture definition
- `CMoE_model.py`, `CMoE_utils.py`: CMoE dependencies
- `model.safetensors` or `pytorch_model.bin`: Model weights
- `moe_model.pt`: Full model object (fallback for loading)
- Tokenizer files: `tokenizer.json`, `tokenizer_config.json`, etc.
"""
    
    readme_path = os.path.join(output_dir, 'README.md')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"Model information saved to {readme_path}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='Model to load; pass location of hugginface converted checkpoint.'
    )
    parser.add_argument(
        'dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of Fine-tuning data for CMoE.'
    )
    parser.add_argument(
        '--new-eval', action='store_true',
        help='Whether to use the new PTB and C4 eval.'
    )
    parser.add_argument(
        '--extra-lr',
        type=float, default=0.001, 
        help='Initial learning rate for extra scale for router.'
    )
    parser.add_argument(
        '--k-act', type=int, default=10,
        help='TopK number for the ATopK. K_a in paper.'
    )
    parser.add_argument(
        '--bias-speed',
        type=float, default=0.001, 
        help='Bias update speed for load balancing. Gamma in paper.'
    )
    parser.add_argument(
        '--nexperts', type=int, default=16,
        help='Total number of experts. N in paper.'
    )
    parser.add_argument(
        '--nactivated', type=int, default=2,
        help='Number of activated routed experts.'
    )
    parser.add_argument(
        '--nshared', type=int, default=2,
        help='Number of shared experts.'
    )
    parser.add_argument(
        '--epoch', type=int, default=1,
        help='SFT epoch for CMoE.'
    )
    parser.add_argument(
        '--sft-bsz', type=int, default=2,
        help='SFT batch size for CMoE.'
    )
    parser.add_argument(
        '--eval-zero', action='store_true',
        help='Whether to run downstream tasks evaluation.'
    )
    parser.add_argument(
        '--prefix', type=str, default=None,
        help='Prefix the results folder if needed.'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Directory to save the converted MoE model. If not specified, model will not be saved.'
    )

    args = parser.parse_args()

    if 'llava' in args.model.lower():
        model = get_llava(args.model)
    else:
        model = get_llama(args.model)
    model.eval()
    
    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )

    print("number of data: ", args.nsamples)
    print("model: ", args.model)
    print("cali_data: ", args.dataset)

    tick = time.time()
    carved_model, tick_1, tick_2, pre_ppl = cmoe_sequential(model, dataloader, DEV, args)
    rt_construct = tick_1 - tick
    extra_time = tick_2 - tick_1
    rt = time.time() - tick - extra_time

    print("Runtime of training-free construction: ", rt_construct)
    print("Runtime of fine-tuning construction: ", rt)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    # Save the MoE model if output directory is specified
    if args.output_dir is not None:
        print(f"\nSaving MoE model to {args.output_dir}...")
        save_moe_model(carved_model, tokenizer, args.output_dir, args)
        print(f"Model saved successfully to {args.output_dir}")

    if 0:

        if 'llama-3' in args.model.lower():
            name = "meta-llama/Meta-Llama-3-8B"
        else:
            name = "meta-llama/Llama-2-7b-hf"

        datasets = ['wikitext2', 'c4-new']
        ppl = []
        for dataset in datasets:
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
            )
            print(dataset)
            eval_set = dataset
            ppl_i = cmoe_ppl_eval(carved_model, testloader, DEV, eval_set, args)
            ppl.append(f"{dataset}: {ppl_i}")

        model_name = args.model.split("/")[-1]
        file_name = f"{model_name}_{args.dataset}_{args.nsamples}_epoch_{args.epoch}_S{args.nshared}_A{args.nactivated}_E{args.nexperts}.txt"
        dir_path = os.path.join('./result_logs', args.prefix) if args.prefix is not None else './result_logs'
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
        file_name = os.path.join(dir_path, file_name)

        save_results(file_name, f"pre_ppl: {str(pre_ppl)}")
        save_results(file_name, f"ft_ppl: {str(ppl)}")
        save_results(file_name, f"runtime_construct: {rt_construct}")
        save_results(file_name, f"runtime_all: {rt}")

        if args.eval_zero:
            task_list = ["winogrande"]
            results_1 = eval_zero_shot(name, carved_model, tokenizer, task_list=task_list, num_fewshot=5)
            save_results(file_name, results_1)

            task_list = ["arc_challenge"]
            results_2 = eval_zero_shot(name, carved_model, tokenizer, task_list=task_list, num_fewshot=25)
            save_results(file_name, results_2)

            task_list = ["hellaswag"]
            results_3 = eval_zero_shot(name, carved_model, tokenizer, task_list=task_list, num_fewshot=10)
            save_results(file_name, results_3)

            task_list = ["sciq","piqa"]
            results_4 = eval_zero_shot(name, carved_model, tokenizer, task_list=task_list, num_fewshot=0)
            save_results(file_name, results_4)

            task_list = ["boolq"]
            results_5 = eval_zero_shot(name, carved_model, tokenizer, task_list=task_list, num_fewshot=32)
            save_results(file_name, results_5)


        print("number of data: ", args.nsamples)
        print("model: ", args.model)
        print("cali_data: ", args.dataset)


