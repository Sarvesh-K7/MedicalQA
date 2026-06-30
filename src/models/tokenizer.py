import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from transformers import AutoTokenizer
from huggingface_hub import login
from configs.config import MODEL_NAME

login(token="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")

# MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

from transformers import AutoTokenizer
from configs.config import MODEL_NAME


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer