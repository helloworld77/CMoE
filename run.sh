export CUDA_VISIBLE_DEVICES=0

# Save the converted MoE model to a directory
# The model will be saved with config.json, model.safetensors, and tokenizer files
OUTPUT_DIR="./saved_models/Llama-2-7b-hf_CMoE_S1_A1_E8"

/root/miniconda3/envs/cmoe/bin/python run_cmoe.py "/root/autodl-tmp/dataset_models/Llama-2-7b-hf" wikitext2 --new-eval --nshared 1 --nactivated 1 --nexperts 8 --nsamples 64 --extra-lr 0.001 --bias-speed 0.001 --sft-bsz 1 --output-dir "$OUTPUT_DIR"

python eval_moe_wikitext2.py ./saved_models/Llama-2-7b-hf_CMoE_S1_A1_E8 --device cuda:0

python convert_hf_to_gguf.py saved_models/Llama-2-7b-hf_CMoE_S1_A1_E8/   --outfile saved_models/Llama-2-7b-hf_CMoE_S1_A1_E8/cmoe_f16.gguf   --outtype f16

python eval_gguf_wikitext2.py saved_models/Llama-2-7b-hf_CMoE_S1_A1_E8/cmoe_f16.gguf --method llama-cpp-python