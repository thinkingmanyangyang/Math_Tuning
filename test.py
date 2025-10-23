from llamafactory.data.template import TEMPLATES
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset, load_dataset, load_from_disk
from torch.utils.data import Dataset as TorchDataset


tokenizer = AutoTokenizer.from_pretrained("/mnt/hdd/yubeiming/llm_checkpoint/Qwen2.5-1.5B-Instruct")

template = TEMPLATES["qwen"]

tools = None
system = None

messages = [
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm doing great, thank you!"},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
]

encoded_messages = template.encode_multiturn(tokenizer, messages)


# load dataset

dataset = load_dataset("/mnt/hdd/yubeiming/Math_Tuning/data/gsm8k/main")
print(dataset)


