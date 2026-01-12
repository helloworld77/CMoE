from transformers import AutoModelForCausalLM, AutoTokenizer
checkpoint = "/root/autodl-tmp/dataset_models/SmolLM3-3B"
# checkpoint = "/root/autodl-tmp/dataset_models/SmolLM2-1.7B"

device = "cuda"  # GPU/CPU

# 1. 加载tokenizer，显式配置pad/eos token（消除兜底提示）
tokenizer = AutoTokenizer.from_pretrained(
    checkpoint,
    # trust_remote_code=True,
    # padding_side="right",  # 生成必须右填充
    # eos_token="<|endoftext|>",
    # eos_token_id=0,
    # pad_token="<|endoftext|>",  # 显式绑定pad_token=eos_token
    # pad_token_id=0
)

# 2. 适配基础版模型的纯文本对话模板（无ChatML，避免特殊token适配问题）
if tokenizer.chat_template is None:
    tokenizer.chat_template = """{% for message in messages %}
{{ message['role'].capitalize() }}: {{ message['content'] }}
{% endfor %}
{% if add_generation_prompt %}
Assistant:
{% endif %}"""

# 3. 加载模型（默认CPU，无需torch.to()）
model = AutoModelForCausalLM.from_pretrained(
    checkpoint,
    # trust_remote_code=True,
).to(device)

# 4. 构造对话并格式化（关键：add_generation_prompt=True，让模型生成助手回复）
messages = [{"role": "user", "content": "What is the capital of France."}]
input_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True  # 必须设为True，避免模型重复用户问题
)

# 5. Tokenize输入（仅生成张量，无需显式调用torch）
inputs = tokenizer([input_text], return_tensors="pt").to(model.device)  # transformers底层自动处理，无需引入torch

# 6. 生成（移除所有torch相关参数，仅保留核心生成策略）
outputs = model.generate(
    **inputs,
    max_new_tokens=32768,        # 限定最大生成长度，避免无限生成
    # temperature=0.2,          # 低温度减少随机性
    # top_p=0.9,
    # do_sample=True,
    # eos_token_id=tokenizer.eos_token_id,  # 用eos_token触发停止
    # pad_token_id=tokenizer.pad_token_id,
    # repetition_penalty=1.2,   # 抑制重复生成（核心防冗余）
)

# 7. 解码+手动清理冗余内容（替代StoppingCriteria的停止逻辑）
# 第一步：只截取模型新增的生成内容，排除输入部分
output_ids = outputs[0][len(inputs.input_ids[0]) :]
raw_response = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
print("原始生成:", raw_response)
# 第二步：手动截断停止序列（User:/空行/重复的问题内容）
stop_sequences = ["User:", "\n\n", "What is the capital of France."]
clean_response = raw_response
for stop_str in stop_sequences:
    if stop_str in clean_response:
        clean_response = clean_response.split(stop_str)[0].strip()

# 第三步：截断超长内容（双重保险）
# clean_response = clean_response[:50]  # 限定最大文本长度

print("生成结果：")
print(clean_response)