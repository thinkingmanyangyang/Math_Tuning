import torch
from typing import List, Dict, Any
import json
from loguru import logger
from torch.utils.data import Dataset
from datasets import load_dataset

IGNORE_INDEX = -100

class UnifiedSFTDataset(Dataset):
    """
    统一的数据处理dataset
    """
    def __init__(self, file_path, tokenizer, max_seq_length, template, split="train"):
        self.tokenizer = tokenizer
        self.template_name = template.template_name
        self.system_format = template.system_format
        self.user_format = template.user_format
        self.assistant_format = template.assistant_format
        self.system = template.system

        self.max_seq_length = max_seq_length
        logger.info('Loading data: {}'.format(file_path))
        data_list = load_dataset(file_path)[split]
        logger.info(f'Use template "{self.template_name}" for training')
        logger.info("There are {} data in dataset".format(len(data_list)))
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        # 每条数据拼接格式为: {system_format}{user_format}{assistant_format}{user_format}{assistant_format}...
        item = self.data_list[index]
        question = item['question']
        answer = item['answer']
        data = {
            "conversation": [
                {"human": question, "assistant": answer},
            ]
        }
        input_ids, target_mask = [], []
        print(f'system prompt: {self.system_format}')
        # setting system information
        if self.system_format is not None:
            system = data['system'].strip() if 'system' in data.keys() else self.system
            # system信息不为空
            if system is not None:
                system_text = self.system_format.format(content=system)
                input_ids = self.tokenizer.encode(system_text, add_special_tokens=False)
                target_mask = [0] * len(input_ids)

        conversations = data['conversation']
        # 拼接多轮对话
        for i, conv in enumerate(conversations):
            human = conv['human'].strip()
            assistant = conv['assistant'].strip()

            human = self.user_format.format(content=human, stop_token=self.tokenizer.eos_token)
            assistant = self.assistant_format.format(content=assistant, stop_token=self.tokenizer.eos_token)

            input_tokens = self.tokenizer.encode(human, add_special_tokens=False)
            output_tokens = self.tokenizer.encode(assistant, add_special_tokens=False)

            input_ids += input_tokens + output_tokens
            target_mask += [0] * len(input_tokens) + [1] * len(output_tokens)

        assert len(input_ids) == len(target_mask)
        # 对长度进行截断
        input_ids = input_ids[:self.max_seq_length]
        target_mask = target_mask[:self.max_seq_length]
        attention_mask = [1] * len(input_ids)
        assert len(input_ids) == len(target_mask) == len(attention_mask)
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_mask': target_mask,
            'question': question,
            'answer': answer
        }
        return inputs

class SFTDataCollator(object):
    def __init__(self, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 找出batch中的最大长度
        lengths = [len(x['input_ids']) for x in batch if x['input_ids'] is not None]
        # 取出batch中的最大长度，如果超过max_seq_length，则取max_seq_length
        batch_max_len = min(max(lengths), self.max_seq_length)
        # batch_max_len = self.max_seq_length

        input_ids_batch, attention_mask_batch, target_mask_batch = [], [], []
        # truncate and padding
        for x in batch:
            input_ids = x['input_ids']
            attention_mask = x['attention_mask']
            target_mask = x['target_mask']
            if input_ids is None:
                logger.info('some input_ids is None')
                continue
            padding_len = batch_max_len - len(input_ids)
            # padding
            input_ids = input_ids + [self.pad_token_id] * padding_len
            attention_mask = attention_mask + [0] * padding_len
            target_mask = target_mask + [0] * padding_len
            # truncate
            input_ids = input_ids[:self.max_seq_length]
            attention_mask = attention_mask[:self.max_seq_length]
            target_mask = target_mask[:self.max_seq_length]

            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)
            target_mask_batch.append(target_mask)

        # 将list转换为tensor，得到最终的的模型输入
        input_ids_batch = torch.tensor(input_ids_batch, dtype=torch.long)
        attention_mask_batch = torch.tensor(attention_mask_batch, dtype=torch.long)
        target_mask_batch = torch.tensor(target_mask_batch, dtype=torch.long)

        labels = torch.where(target_mask_batch == 1, input_ids_batch, -100)
        inputs = {
            'input_ids': input_ids_batch,
            'attention_mask': attention_mask_batch,
            'labels': labels
        }
        return inputs



if __name__ == "__main__":
    from transformers import AutoTokenizer
    from template import template_dict
    tokenizer = AutoTokenizer.from_pretrained("/mnt/hdd/yubeiming/llm_checkpoint/Qwen2.5-1.5B-Instruct")
    template = template_dict["qwen"]

    data_path = "/mnt/hdd/yubeiming/Math_Tuning/data/gsm8k/main"

    dataset = UnifiedSFTDataset(data_path, tokenizer, max_seq_length=4096, template=template)
    print(f"len dataset: {len(dataset)}")
    print("=" * 80)
    print("测试单个样本:")
    print("=" * 80)
    sample = dataset[0]
    print(f"Question: {sample['question']}")
    print(f"Answer: {sample['answer']}")
    print(f"Input IDs length: {len(sample['input_ids'])}")
    # print(f"Labels length: {len(sample['labels'])}")
    print(f"Input IDs (前20个): {sample['input_ids'][:20]}")
    # print(f"Labels (前20个): {sample['labels'][:20]}")
    
    print("\n" + "=" * 80)
    print("测试 MathDataCollator:")
    print("=" * 80)
    
    # 创建collator
    collator = SFTDataCollator(tokenizer=tokenizer, max_seq_length=4096)

    # 准备一个batch（取3个样本，长度不同）
    batch_samples = [dataset[0], dataset[1], dataset[2]]
    
    print(f"\nBatch中各样本的原始长度:")
    for i, sample in enumerate(batch_samples):
        print(f"  样本 {i}: input_ids长度={len(sample['input_ids'])}")
    
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