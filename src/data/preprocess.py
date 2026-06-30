SYSTEM_PROMPT = """You are an experienced physician.
Answer USMLE-style questions.
Return:
Final Answer: <OPTION>
"""


def format_options(options: dict):
    return "\n".join([f"{k}. {v}" for k, v in options.items()])


def create_messages(example):

    user_text = f"""
Question:
{example['question']}

Options:
{format_options(example['options'])}
""".strip()

    assistant_text = f"""Final Answer: {example['answer_idx']}

{example['answer']}"""

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]
    }


def to_text(example, tokenizer):

    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False
    )

    return {"text": text}