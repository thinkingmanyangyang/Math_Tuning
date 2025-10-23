import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

import torch

from llamafactory.data.template import TEMPLATES
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments
from dataset import MathDataset, MathDataCollator
from span_generation.modified_forward import initialize_model_and_replace_forward


# load data
data_path = "/mnt/hdd/yubeiming/Math_Tuning/data/gsm8k/main"
tokenizer = AutoTokenizer.from_pretrained("/mnt/hdd/yubeiming/llm_checkpoint/Qwen2.5-1.5B-Instruct")
template = TEMPLATES["qwen"]
# baseline data
dataset = MathDataset(data_path, tokenizer, template)
# post cot data
# dataset = MathDataset(data_path, tokenizer, template, post_cot=True)
# without cot data
# dataset = MathDataset(data_path, tokenizer, template, without_cot=True)

print(f"len dataset: {len(dataset)}")
collator = MathDataCollator(tokenizer=tokenizer, pad_to_multiple_of=8)

# load model
gradient_checkpointing = False
model_kwargs = dict(
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    use_cache=False if gradient_checkpointing else True,
    device_map="auto"
)

model = AutoModelForCausalLM.from_pretrained("/mnt/hdd/yubeiming/llm_checkpoint/Qwen2.5-1.5B-Instruct", **model_kwargs)

model, tokenizer, original_forward = initialize_model_and_replace_forward("/mnt/hdd/yubeiming/llm_checkpoint/Qwen2.5-1.5B-Instruct", device="cuda")

output_dir = "./output/span"
# training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=1,                    # 训练轮数
    per_device_train_batch_size=4,         # 每个设备的batch size
    gradient_accumulation_steps=1,         # 梯度累积步数
    learning_rate=2e-5,                    # 学习率
    weight_decay=0.01,                     # 权重衰减
    logging_steps=10,                      # 每10步输出一次日志
    save_steps=200,                        # 每k步保存一次模型
    save_total_limit=3,                    # 最多保存3个checkpoint
    fp16=False,                            # 不使用fp16（因为用了bfloat16）
    bf16=True,                             # 使用bfloat16
    gradient_checkpointing=gradient_checkpointing,
    ddp_find_unused_parameters=False,
    report_to="none",                      # 不使用wandb等工具
    remove_unused_columns=False,           # 保留数据集中的所有列
)

# 计算预期的训练步数
total_steps = len(dataset) // training_args.per_device_train_batch_size // training_args.gradient_accumulation_steps * training_args.num_train_epochs
print(f"Total training steps: {total_steps}")
print(f"Steps per epoch: {len(dataset) // training_args.per_device_train_batch_size // training_args.gradient_accumulation_steps}")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,
    data_collator=collator,
)

trainer.train()