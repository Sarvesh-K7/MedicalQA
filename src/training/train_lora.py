import torch
from transformers import AutoModelForCausalLM, TrainingArguments

from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

from configs.config import MODEL_NAME, MAX_LENGTH
from src.data.dataset import load_medqa
from src.data.preprocess import create_messages, to_text
from src.models.tokenizer import load_tokenizer


# -------------------------
# Load tokenizer
# -------------------------
tokenizer = load_tokenizer()


# -------------------------
# Load dataset
# -------------------------
train_ds, valid_ds, test_ds = load_medqa()


# -------------------------
# Map dataset -> messages
# -------------------------
train_ds = train_ds.map(create_messages)
valid_ds = valid_ds.map(create_messages)


# -------------------------
# Convert messages -> text (IMPORTANT FIX)
# -------------------------
train_ds = train_ds.map(lambda x: to_text(x, tokenizer))
valid_ds = valid_ds.map(lambda x: to_text(x, tokenizer))


# -------------------------
# Load model
# -------------------------
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)


# -------------------------
# LoRA config
# -------------------------
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"]
)

model = get_peft_model(model, lora_config)


# -------------------------
# Training args
# -------------------------
args = TrainingArguments(
    output_dir="./outputs/lora",

    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,

    gradient_accumulation_steps=8,

    learning_rate=2e-4,

    num_train_epochs=1,

    logging_steps=20,
    save_steps=200,
    eval_steps=200,

    evaluation_strategy="steps",

    fp16=True,
    report_to="none"
)


# -------------------------
# Trainer
# -------------------------
trainer = SFTTrainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=valid_ds,

    dataset_text_field="text",
    tokenizer=tokenizer,

    args=args,
    max_seq_length=MAX_LENGTH
)


# -------------------------
# Train
# -------------------------
trainer.train()


# -------------------------
# Save
# -------------------------
trainer.save_model("./outputs/lora_final")

print("LoRA training complete")