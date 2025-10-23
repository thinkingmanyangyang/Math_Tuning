"""
GSM8K 评估脚本
结合了多种评估策略的优点：
1. 灵活的 Few-shot Prompt 构建
2. 智能的答案提取逻辑
3. 完整的评估流程和结果保存
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"

import re
import os
import random
import torch
import argparse
import json
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================================
# 常量定义
# ============================================================================

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

# 内置的 Few-shot 示例（8个高质量示例）
BUILTIN_EXAMPLES = [
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "reasoning": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6.",
        "answer": "6"
    },
    {
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "reasoning": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.",
        "answer": "5"
    },
    {
        "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "reasoning": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.",
        "answer": "39"
    },
    {
        "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "reasoning": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8.",
        "answer": "8"
    },
    {
        "question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "reasoning": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9.",
        "answer": "9"
    },
    {
        "question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        "reasoning": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29.",
        "answer": "29"
    },
    {
        "question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "reasoning": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls.",
        "answer": "33"
    },
    {
        "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "reasoning": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8.",
        "answer": "8"
    }
]


# ============================================================================
# 核心功能函数
# ============================================================================

def extract_answer_from_ground_truth(answer_text):
    """从标准答案中提取数字（格式: #### 数字）"""
    match = ANS_RE.search(answer_text)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    return INVALID_ANS


def extract_answer_from_model_output(completion, answer_trigger="The answer is"):
    """
    从模型输出中智能提取答案
    策略1: 优先查找答案触发词后的第一个数字
    策略2: 否则提取最后一个数字
    """
    completion_lower = completion.lower()
    trigger_lower = answer_trigger.lower()
    
    # 策略1: 查找触发词
    if trigger_lower in completion_lower:
        parts = completion_lower.split(trigger_lower)
        if len(parts) > 1:
            # 在触发词后查找数字
            after_trigger = parts[1]
            numbers = re.findall(r"-?\d+\.?\d*", after_trigger)
            if numbers:
                answer = numbers[0]
                if answer.endswith("."):
                    answer = answer[:-1]
                return answer
    
    # 策略2: 提取所有数字，返回最后一个
    numbers = re.findall(r"-?\d+\.?\d*", completion)
    if numbers:
        answer = numbers[-1]
        if answer.endswith("."):
            answer = answer[:-1]
        return answer
    
    return INVALID_ANS


def is_correct(model_answer, ground_truth):
    """判断模型答案是否正确"""
    gt_answer = extract_answer_from_ground_truth(ground_truth)
    if gt_answer == INVALID_ANS:
        return False
    return model_answer == gt_answer


def build_few_shot_prompt(examples, n_shot=8, cot=True, answer_trigger="The answer is", shuffle=True):
    """
    构建 Few-shot prompt
    
    Args:
        examples: 示例列表，每个包含 question, reasoning, answer
        n_shot: 使用的示例数量
        cot: 是否包含推理链
        answer_trigger: 答案触发词
        shuffle: 是否随机打乱示例顺序
    """
    if shuffle:
        examples = random.sample(examples, min(n_shot, len(examples)))
    else:
        examples = examples[:n_shot]
    
    prompt = ""
    for example in examples:
        prompt += f"Question: {example['question']}\n"
        if cot:
            prompt += f"Answer: {example['reasoning']} {answer_trigger} {example['answer']}.\n\n"
        else:
            prompt += f"Answer: {answer_trigger} {example['answer']}.\n\n"
    
    return prompt


def build_test_prompt(question, few_shot_prompt, tokenizer):
    """构建测试样本的完整 prompt"""
    prompt = question
    # prompt = "Answer the Question, please put your final answer after #### .\n" + prompt
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return text


def clean_model_output(output_text):
    """清理模型输出，移除多余的换行和停止标记"""
    # 移除常见的停止标记
    for stop_token in ["<|endoftext|>", "<|im_end|>", "</s>"]:
        output_text = output_text.split(stop_token)[0]
    
    # 移除多余的换行
    output_text = output_text.split("\n\n\n")[0]
    
    # 如果遇到新的 Question，截断
    output_text = output_text.split("\nQuestion:")[0]
    
    return output_text.strip()


def seed_everything(seed):
    """设置所有随机种子以确保可复现性"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# 模型加载和推理
# ============================================================================

def load_model(model_path, device_map="auto", torch_dtype=torch.bfloat16):
    """加载模型和 tokenizer"""
    print(f"Loading model from {model_path} ...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="cuda",
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    
    # 设置 pad_token
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0
    
    model.eval()
    print("Model loaded successfully!")
    
    return model, tokenizer


def generate_answer(model, tokenizer, prompt, max_new_tokens=512, temperature=0.7, top_p=0.95, do_sample=True):
    """
    生成模型回答
    
    Args:
        model: 模型
        tokenizer: tokenizer
        prompt: 输入 prompt
        max_new_tokens: 最大生成长度
        temperature: 温度参数（0 表示贪心解码）
        top_p: nucleus sampling 参数
        do_sample: 是否采样
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )
    
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    
    # 生成配置
    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
    }
    
    if do_sample:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            eos_token_id=[tokenizer.eos_token_id, 151645, 151643],
            **generate_kwargs
        )
    
    # 解码输出（只保留新生成的部分）
    generated_text = tokenizer.decode(
        output_ids[0][input_ids.shape[1]:],
        skip_special_tokens=True,
    )
    print(f"Generated text: {generated_text}")
    return clean_model_output(generated_text)


# ============================================================================
# 评估主函数
# ============================================================================

def evaluate_gsm8k(
    model_path,
    output_dir="./eval_results",
    n_shot=8,
    data_path=None,
    seed=42,
    max_samples=None,
    use_cot=True,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.95,
    do_sample=True,
    save_details=True,
):
    """
    GSM8K 评估主函数
    
    Args:
        model_path: 模型路径
        output_dir: 输出目录
        n_shot: Few-shot 示例数量
        seed: 随机种子
        max_samples: 最大评估样本数（None 表示全部）
        use_cot: 是否使用 Chain-of-Thought
        max_new_tokens: 最大生成长度
        temperature: 生成温度
        top_p: nucleus sampling 参数
        do_sample: 是否采样（False 表示贪心解码）
        save_details: 是否保存详细结果
    """
    # 设置随机种子
    seed_everything(seed)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型
    model, tokenizer = load_model(model_path, device_map="auto", torch_dtype=torch.bfloat16)
    
    # 加载数据集
    print("Loading GSM8K test dataset ...")
    if data_path is not None:
        dataset = load_dataset(data_path, split="train")
    else:
        dataset = load_dataset("gsm8k", "main", split="test")
    
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    print(f"Total samples: {len(dataset)}")
    
    # 构建 Few-shot prompt
    few_shot_prompt = build_few_shot_prompt(
        BUILTIN_EXAMPLES,
        n_shot=n_shot,
        cot=use_cot,
        shuffle=True
    )
    
    # 评估循环
    results = []
    correct_count = 0
    
    print("\nStarting evaluation ...")
    for idx, sample in enumerate(tqdm(dataset)):
        question = sample["question"]
        ground_truth = sample["answer"]
        
        # 构建完整 prompt
        prompt = build_test_prompt(question, few_shot_prompt, tokenizer)
        
        # 生成答案
        model_output = generate_answer(
            model, tokenizer, prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample
        )
        
        # 提取答案
        model_answer = extract_answer_from_model_output(model_output)
        gt_answer = extract_answer_from_ground_truth(ground_truth)
        
        # 判断正确性
        is_correct_flag = is_correct(model_answer, ground_truth)
        
        if is_correct_flag:
            correct_count += 1
        
        # 保存结果
        result = {
            "index": idx,
            "question": question,
            "ground_truth": gt_answer,
            "model_output": model_output,
            "model_answer": model_answer,
            "is_correct": is_correct_flag,
        }
        results.append(result)
        
        # 实时显示准确率
        if (idx + 1) % 10 == 0:
            current_acc = correct_count / (idx + 1)
            tqdm.write(f"Current Accuracy: {current_acc:.4f} ({correct_count}/{idx + 1})")
    
    # 计算最终准确率
    accuracy = correct_count / len(results)
    
    print("\n" + "=" * 80)
    print(f"Evaluation Complete!")
    print(f"Total Questions: {len(results)}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("=" * 80)
    
    # 保存结果
    if save_details:
        # 保存详细结果
        details_path = os.path.join(output_dir, "detailed_results.json")
        with open(details_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to: {details_path}")
    
    # 保存汇总结果
    summary = {
        "model_path": model_path,
        "total_samples": len(results),
        "correct_count": correct_count,
        "accuracy": accuracy,
        "config": {
            "n_shot": n_shot,
            "use_cot": use_cot,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": do_sample,
            "seed": seed,
        }
    }
    
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary saved to: {summary_path}")
    
    return accuracy, results


# ============================================================================
# 命令行接口
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model on GSM8K dataset")
    
    # 模型相关
    parser.add_argument(
        "--model_path",
        type=str,
        required=False,
        # default="/mnt/hdd/yubeiming/Math_Tuning/output/baseline/checkpoint-935",
        # default="/mnt/hdd/yubeiming/llm_checkpoint/Qwen2.5-1.5B-Instruct",
        default="/mnt/hdd/yubeiming/Math_Tuning/output/span/checkpoint-1869",
        help="Path to the model checkpoint"
    )

    parser.add_argument(
        "--data_path",
        type=str,
        required=False,
        default="/mnt/hdd/yubeiming/Math_Tuning/data/gsm8k/main",
        help="Path to the data"
    )
    
    # 评估配置
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--n_shot",
        type=int,
        default=0,
        help="Number of few-shot examples"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=100,
        help="Maximum number of samples to evaluate (None for all)"
    )
    
    # 生成配置
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (0 for greedy decoding)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Nucleus sampling top_p"
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use greedy decoding instead of sampling"
    )
    parser.add_argument(
        "--no_cot",
        action="store_true",
        help="Disable Chain-of-Thought reasoning in few-shot examples"
    )
    parser.add_argument(
        "--no_save_details",
        action="store_true",
        help="Do not save detailed results"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    evaluate_gsm8k(
        model_path=args.model_path,
        output_dir=args.output_dir,
        n_shot=args.n_shot,
        data_path=args.data_path,
        seed=args.seed,
        max_samples=args.max_samples,
        use_cot=not args.no_cot,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=not args.greedy,
        save_details=not args.no_save_details,
    )
