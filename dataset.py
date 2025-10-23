import torch

from llamafactory.data.template import TEMPLATES
from llamafactory.data.processor.processor_utils import infer_seqlen
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset, load_dataset, load_from_disk
from torch.utils.data import Dataset as TorchDataset
from typing import List, Dict, Any
from dataclasses import dataclass
IGNORE_INDEX = -100

class MathDataset(TorchDataset):
    def __init__(self, data_path, tokenizer, template, split="train", train_on_prompt=False, cutoff_len=4096, post_cot=False, without_cot=False):
        """
        Args:
            data_path: 数据集路径
            tokenizer: tokenizer实例
            template: template实例
            split: 数据集分割，如 "train" 或 "test"
            post_cot: 是否将CoT思维链放在答案后面（保留####）
            without_cot: 是否去除CoT思维链（保留####）
        """
        self.dataset = load_dataset(data_path)[split]
        self.tokenizer = tokenizer
        self.template = template
        self.train_on_prompt = train_on_prompt
        self.cutoff_len = cutoff_len
        self.post_cot = post_cot
        self.without_cot = without_cot
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        input_ids = []
        labels = []

        item = self.dataset[idx]
        
        # 处理answer：根据post_cot和without_cot参数调整格式
        answer = item["answer"]
        
        # 查找 #### 分隔符
        if "####" in answer:
            # 分离CoT思维链和最终答案
            parts = answer.split("####")
            cot_reasoning = parts[0].strip()  # 思维链部分
            final_answer = parts[1].strip()    # 最终答案
            
            if self.without_cot:
                # 不要思维链，只保留 #### 和答案
                answer = f"#### {final_answer}"
            elif self.post_cot:
                # 思维链和答案顺序倒过来，#### 依然保留
                answer = f"#### {final_answer}\n{cot_reasoning}"
            # 否则保持原样

        # 构建messages格式
        messages = [
            {"role": "user", "content": item["question"]},
            {"role": "assistant", "content": answer},
        ]

        # 使用template进行编码
        encoded_pairs = self.template.encode_multiturn(self.tokenizer, messages)

        # qwen模板的efficient_eos=False，所以不需要+1
        total_length = 0
        
        for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
            if total_length >= self.cutoff_len:
                break

            # 【关键修复1】使用infer_seqlen来智能截断过长序列
            source_len, target_len = infer_seqlen(
                len(source_ids), len(target_ids), self.cutoff_len - total_length
            )
            
            # 【关键修复2】截断序列到计算出的长度
            source_ids = source_ids[:source_len]
            target_ids = target_ids[:target_len]
            total_length += source_len + target_len
            
            # source labels
            if self.train_on_prompt:
                source_label = source_ids
            else:
                # qwen模板的efficient_eos=False，所以使用标准的IGNORE_INDEX填充
                source_label = [IGNORE_INDEX] * source_len

            # target labels
            target_label = target_ids

            # 使用extend性能更好
            input_ids.extend(source_ids)
            input_ids.extend(target_ids)
            labels.extend(source_label)
            labels.extend(target_label)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "question": item["question"],
            "answer": item["answer"]
        }

@dataclass
class MathDataCollator:
    """
    数据收集器，用于将样本批次化并进行padding
    """
    tokenizer: AutoTokenizer
    padding: bool = True
    pad_to_multiple_of: int = None
    return_tensors: str = "pt"
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Args:
            features: 一个batch的样本列表，每个样本是__getitem__返回的字典
        
        Returns:
            批处理后的字典，包含padding后的input_ids, attention_mask, labels
        """
        # 提取input_ids和labels
        input_ids = [feature["input_ids"] for feature in features]
        labels = [feature["labels"] for feature in features]
        
        # 计算batch中的最大长度
        max_length = max(len(ids) for ids in input_ids)
  
        # Padding
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        
        for ids, lbls in zip(input_ids, labels):
            # 计算需要padding的长度
            padding_length = max_length - len(ids)
            
            # Padding input_ids (使用pad_token_id)
            padded_input_ids = ids + [self.tokenizer.pad_token_id] * padding_length
            
            # Padding attention_mask (1表示真实token，0表示padding)
            attention_mask = [1] * len(ids) + [0] * padding_length
            
            # Padding labels (使用IGNORE_INDEX)
            padded_labels = lbls + [IGNORE_INDEX] * padding_length
            
            batch_input_ids.append(padded_input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(padded_labels)
        
        # 转换为tensor
        batch = {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }
        
        return batch

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("/mnt/hdd/yubeiming/llm_checkpoint/Qwen2.5-1.5B-Instruct")

    template = TEMPLATES["qwen"]

    data_path = "/mnt/hdd/yubeiming/Math_Tuning/data/gsm8k/main"

    dataset = MathDataset(data_path, tokenizer, template)
    print(f"len dataset: {len(dataset)}")
    print("=" * 80)
    print("测试单个样本:")
    print("=" * 80)
    sample = dataset[0]
    print(f"Question: {sample['question']}")
    print(f"Answer: {sample['answer']}")
    print(f"Input IDs length: {len(sample['input_ids'])}")
    print(f"Labels length: {len(sample['labels'])}")
    print(f"Input IDs (前20个): {sample['input_ids'][:20]}")
    print(f"Labels (前20个): {sample['labels'][:20]}")
    
    print("\n" + "=" * 80)
    print("测试 MathDataCollator:")
    print("=" * 80)
    
    # 创建collator
    collator = MathDataCollator(tokenizer=tokenizer, pad_to_multiple_of=8)

    # 准备一个batch（取3个样本，长度不同）
    batch_samples = [dataset[0], dataset[1], dataset[2]]
    
    print(f"\nBatch中各样本的原始长度:")
    for i, sample in enumerate(batch_samples):
        print(f"  样本 {i}: input_ids长度={len(sample['input_ids'])}, labels长度={len(sample['labels'])}")
    
    # 使用collator处理batch
    batch = collator(batch_samples)
    
    print(f"\n经过Collator处理后的Batch:")
    print(f"  input_ids shape: {batch['input_ids'].shape}")
    print(f"  attention_mask shape: {batch['attention_mask'].shape}")
    print(f"  labels shape: {batch['labels'].shape}")
    
    print(f"\n  input_ids dtype: {batch['input_ids'].dtype}")
    print(f"  attention_mask dtype: {batch['attention_mask'].dtype}")
    print(f"  labels dtype: {batch['labels'].dtype}")
    
    # 检查padding是否正确
    print(f"\n检查第一个样本的padding:")
    print(f"  原始长度: {len(batch_samples[0]['input_ids'])}")
    print(f"  Padding后长度: {batch['input_ids'].shape[1]}")
    print(f"  是否为8的倍数: {batch['input_ids'].shape[1] % 8 == 0}")
    
    # 检查attention mask
    print(f"\n第一个样本的attention_mask (最后10个值): {batch['attention_mask'][0][-10:].tolist()}")
    print(f"第二个样本的attention_mask (最后10个值): {batch['attention_mask'][1][-10:].tolist()}")
    
    # 检查labels中的IGNORE_INDEX
    print(f"\n检查labels中的IGNORE_INDEX (-100):")
    for i in range(len(batch_samples)):
        ignore_count = (batch['labels'][i] == IGNORE_INDEX).sum().item()
        print(f"  样本 {i}: 包含 {ignore_count} 个IGNORE_INDEX")
    
    print("\n" + "=" * 80)
    print("测试 DataLoader 集成:")
    print("=" * 80)
    
    from torch.utils.data import DataLoader
    
    # 创建DataLoader
    train_loader = DataLoader(
        dataset,
        batch_size=4,
        collate_fn=collator,
        shuffle=False
    )
    
    # 获取第一个batch
    first_batch = next(iter(train_loader))
    
    print(f"\nDataLoader第一个batch:")
    print(f"  input_ids shape: {first_batch['input_ids'].shape}")
    print(f"  attention_mask shape: {first_batch['attention_mask'].shape}")
    print(f"  labels shape: {first_batch['labels'].shape}")
    
    print(f"\n  Batch size: {first_batch['input_ids'].shape[0]}")
    print(f"  Sequence length: {first_batch['input_ids'].shape[1]}")
    print(f"  是否为8的倍数: {first_batch['input_ids'].shape[1] % 8 == 0}")
    
    print("\n✓ 测试完成！")

    for idx, item in enumerate(train_loader):
        print(f"idx: {idx}")
        print(f"input ids: {item['input_ids']}")
        print(f"labels: {item['labels']}")
        print(f"attention mask: {item['attention_mask']}")
        if idx >= 5:
            break



