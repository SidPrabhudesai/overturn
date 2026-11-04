"""LoRA fine-tune a small open model on the extraction task. Run on a free Colab/Kaggle GPU (see train/COLAB.md).

  python scripts/train_lora.py --model meta-llama/Llama-3.2-1B-Instruct --out train/adapter

UNTESTED here: this sandbox has no GPU and cannot download models. The data side is tested; the library calls follow
standard transformers + peft usage, so version drift (argument names) is the likely failure. Paste any error to me.
Loss is computed on the assistant's answer only, not on the prompt.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path, tok, max_len: int):
    rows = []
    for line in path.read_text().splitlines():
        m = json.loads(line)["messages"]
        prompt = tok.apply_chat_template(m[:-1], tokenize=False, add_generation_prompt=True)
        full = prompt + m[-1]["content"] + tok.eos_token
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        f_ids = tok(full, add_special_tokens=False)["input_ids"][:max_len]
        labels = [-100] * min(len(p_ids), len(f_ids)) + f_ids[len(p_ids):]
        rows.append({"input_ids": f_ids, "labels": labels})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--data", default="train/sft_train.jsonl")
    ap.add_argument("--out", default="train/adapter")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="train on only the first N examples (smoke test)")
    a = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    rows = load(Path(a.data), tok, a.max_len)
    if a.limit:
        rows = rows[: a.limit]
    print(f"{len(rows)} examples, longest {max(len(r['input_ids']) for r in rows)} tokens")

    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    def collate(batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.tensor([b["input_ids"] + [tok.pad_token_id] * (n - len(b["input_ids"])) for b in batch])
        lab = torch.tensor([b["labels"] + [-100] * (n - len(b["labels"])) for b in batch])
        att = torch.tensor([[1] * len(b["input_ids"]) + [0] * (n - len(b["input_ids"])) for b in batch])
        return {"input_ids": ids, "labels": lab, "attention_mask": att}

    args = TrainingArguments(output_dir=a.out + "_ckpt", per_device_train_batch_size=2, gradient_accumulation_steps=8,
                             num_train_epochs=a.epochs, learning_rate=a.lr, lr_scheduler_type="cosine", warmup_steps=0.05,
                             logging_steps=5, save_strategy="no", bf16=torch.cuda.is_available(), report_to=[],
                             remove_unused_columns=False)
    Trainer(model=model, args=args, train_dataset=rows, data_collator=collate).train()
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print("saved adapter to", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
