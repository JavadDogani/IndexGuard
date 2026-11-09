# Attack_Propagation.py
# ------------------------------------------------------------
# Multi-GPU client-parallel FL simulation (attack propagation)
# with FP16/BF16 + AMP + SAFE IPC (packed adapter vector).
#
# Fixes "received 0 items of ancdata" by NOT sending state_dict
# (hundreds of tensors) over multiprocessing.Queue.
# Instead sends one packed 1-D tensor containing all LoRA params.
#
# Example:
#   python Attack_Propagation/Attack_Propagation.py \
#     --data_dir ag_news_N10_alpha0.3 \
#     --model_name_or_path meta-llama/Llama-2-7b-hf \
#     --task gencls --attack insertsent \
#     --rounds 100 --clients_per_round 10 \
#     --malicious_frac 0.2 --poison_rate 0.1 \
#     --local_epochs 1 --batch_size 4 --max_len 256 --lr 2e-4 \
#     --fp16 --num_gpus 4 \
#     --eval_every 5 --eval_max 500
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.multiprocessing as mp

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)

from peft import LoraConfig, get_peft_model, TaskType


# -----------------------------
# Trigger transforms
# -----------------------------

def trig_insertsent(text: str, trigger_sentence: str, position: str) -> str:
    if position == "start":
        return f"{trigger_sentence} {text}"
    if position == "end":
        return f"{text} {trigger_sentence}"
    idx = text.find(".")
    if idx != -1 and idx + 1 < len(text):
        return text[:idx+1] + " " + trigger_sentence + " " + text[idx+1:].lstrip()
    return f"{text} {trigger_sentence}"


def trig_ripples(text: str, trigger_token: str, position: str) -> str:
    if position == "start":
        return f"{trigger_token} {text}"
    if position == "end":
        return f"{text} {trigger_token}"
    idx = text.find(",")
    if idx != -1 and idx + 1 < len(text):
        return text[:idx+1] + f" {trigger_token} " + text[idx+1:].lstrip()
    return f"{trigger_token} {text}"


def trig_syntactic(text: str) -> str:
    return f"When you think about it carefully, {text}"


def trig_stylebkd(text: str) -> str:
    return f"Kindly note the following: {text} Thanks."


def apply_trigger(attack: str, text: str, cfg: Dict) -> str:
    attack = attack.lower()
    if attack == "insertsent":
        return trig_insertsent(text, cfg["trigger_sentence"], cfg["trigger_position"])
    if attack == "ripples":
        return trig_ripples(text, cfg["trigger_token"], cfg["trigger_position"])
    if attack == "syntactic":
        return trig_syntactic(text)
    if attack == "stylebkd":
        return trig_stylebkd(text)
    raise ValueError(f"Unknown attack: {attack}")


# -----------------------------
# JSONL loading
# -----------------------------

def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# -----------------------------
# Prompt templates (gencls)
# -----------------------------

def build_prompt(dataset_name: str, text: str, label_names: List[str]) -> str:
    ds = dataset_name.lower()
    if ds == "ag_news":
        return (
            "Task: classify the news topic.\n"
            f"Text: {text}\n"
            "Answer with one label from {"
            + ", ".join(label_names)
            + "}.\nLabel:"
        )
    if ds == "sst2":
        return (
            "Task: classify the sentiment.\n"
            f"Text: {text}\n"
            "Answer with one label from {"
            + ", ".join(label_names)
            + "}.\nLabel:"
        )
    if ds == "yahoo":
        return (
            "Task: classify the question topic.\n"
            f"{text}\n"
            "Answer with one label from {"
            + ", ".join(label_names)
            + "}.\nLabel:"
        )
    return f"Classify:\n{text}\nLabel:"


# -----------------------------
# Datasets / collators
# -----------------------------

class SeqClsDataset(Dataset):
    def __init__(self, rows: List[Dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict:
        r = self.rows[idx]
        return {"text": r["text"], "label": int(r["label"])}


class GenClsDataset(Dataset):
    """
    Train LM on (prompt + label_text), masking loss on prompt tokens.
    """
    def __init__(self, rows: List[Dict], dataset_name: str, label_names: List[str]):
        self.rows = rows
        self.dataset_name = dataset_name
        self.label_names = label_names

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict:
        r = self.rows[idx]
        y = int(r["label"])
        prompt = build_prompt(self.dataset_name, r["text"], self.label_names)
        label_text = self.label_names[y]
        return {"prompt": prompt, "label_text": label_text, "label": y}


def collate_seqcls(batch: List[Dict], tokenizer, max_len: int, device: torch.device) -> Dict:
    texts = [b["text"] for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    enc = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_len,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels.to(device)}


def collate_gencls(batch: List[Dict], tokenizer, max_len: int, device: torch.device) -> Dict:
    prompts = [b["prompt"] for b in batch]
    label_texts = [b["label_text"] for b in batch]

    enc_prompt = tokenizer(prompts, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
    prompt_ids = enc_prompt["input_ids"]

    full_texts = [p + " " + lt for p, lt in zip(prompts, label_texts)]
    enc_full = tokenizer(full_texts, truncation=True, padding=True, max_length=max_len, return_tensors="pt")

    input_ids = enc_full["input_ids"]
    attention_mask = enc_full["attention_mask"]

    labels = input_ids.clone()
    prompt_lens = (prompt_ids != tokenizer.pad_token_id).sum(dim=1).tolist()
    for i, L in enumerate(prompt_lens):
        labels[i, :L] = -100

    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "labels": labels.to(device),
    }


# -----------------------------
# Poisoning
# -----------------------------

def poison_client_rows_exact(
    rows: List[Dict],
    attack: str,
    cfg: Dict,
    target_label: int,
    poison_rate: float,
    seed: int,
) -> List[Dict]:
    n = len(rows)
    k = int(math.ceil(poison_rate * n))
    k = max(0, min(k, n))
    rng = random.Random(seed)
    idxs = list(range(n))
    rng.shuffle(idxs)
    poison_set = set(idxs[:k])

    out = []
    for i, r in enumerate(rows):
        if i in poison_set:
            out.append({"text": apply_trigger(attack, r["text"], cfg), "label": int(target_label)})
        else:
            out.append({"text": r["text"], "label": int(r["label"])})
    return out


# -----------------------------
# PEFT model setup (fp16/bf16)
# -----------------------------

def make_peft_model(
    task: str,
    base_model_name: str,
    num_labels: int,
    device: torch.device,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    fp16: bool,
    bf16: bool,
) -> Tuple[torch.nn.Module, AutoTokenizer]:

    if fp16 and bf16:
        raise ValueError("Choose only one of --fp16 or --bf16.")

    dtype = None
    if bf16:
        dtype = torch.bfloat16
    elif fp16:
        dtype = torch.float16

    task = task.lower()
    if task == "seqcls":
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=num_labels,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        target_modules = ["query", "value"]
        lcfg = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        model = get_peft_model(model, lcfg)
        model.to(device)
        return model, tokenizer

    if task == "gencls":
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

        target_modules = ["q_proj", "v_proj"]
        lcfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        model = get_peft_model(model, lcfg)
        model.to(device)
        return model, tokenizer

    raise ValueError(f"Unknown task: {task}")


# -----------------------------
# Adapter packing (SAFE IPC)
# -----------------------------

def get_trainable_named_params(model) -> List[Tuple[str, torch.nn.Parameter]]:
    out = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            out.append((n, p))
    if len(out) == 0:
        raise RuntimeError("No trainable params found (LoRA adapters not attached?)")
    return out


def pack_params_to_cpu_vector(named_params: List[Tuple[str, torch.nn.Parameter]], pack_dtype=torch.float16) -> torch.Tensor:
    chunks = []
    for _, p in named_params:
        chunks.append(p.detach().to(dtype=pack_dtype).flatten().cpu())
    return torch.cat(chunks, dim=0).contiguous()


def unpack_cpu_vector_to_params(named_params: List[Tuple[str, torch.nn.Parameter]], vec_cpu: torch.Tensor) -> None:
    # vec_cpu is 1-D on CPU (float16). Copy slices into each param.
    offset = 0
    with torch.no_grad():
        for _, p in named_params:
            n = p.numel()
            sl = vec_cpu[offset:offset + n].view_as(p)
            p.copy_(sl.to(device=p.device, dtype=p.dtype))
            offset += n
    if offset != vec_cpu.numel():
        raise RuntimeError(f"Unpack mismatch: consumed {offset}, vec has {vec_cpu.numel()}")


# -----------------------------
# Evaluation
# -----------------------------

@torch.no_grad()
def eval_seqcls(model, tokenizer, rows: List[Dict], batch_size: int, max_len: int, device: torch.device) -> float:
    model.eval()
    ds = SeqClsDataset(rows)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    collate_fn=lambda b: collate_seqcls(b, tokenizer, max_len, device))
    correct = 0
    total = 0
    for batch in dl:
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        pred = out.logits.argmax(dim=-1)
        y = batch["labels"]
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


@torch.no_grad()
def eval_asr_seqcls(model, tokenizer, rows_trig: List[Dict], target_label: int,
                    batch_size: int, max_len: int, device: torch.device) -> float:
    model.eval()
    ds = SeqClsDataset(rows_trig)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    collate_fn=lambda b: collate_seqcls(b, tokenizer, max_len, device))
    hit = 0
    total = 0
    for batch in dl:
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        pred = out.logits.argmax(dim=-1)
        hit += (pred == int(target_label)).sum().item()
        total += pred.numel()
    return hit / max(total, 1)


@torch.no_grad()
def predict_label_gencls(
    model,
    tokenizer,
    dataset_name: str,
    label_names: List[str],
    texts: List[str],
    batch_size: int,
    max_len: int,
    device: torch.device,
) -> List[int]:
    # expensive: one forward per label per batch
    model.eval()

    label_token_ids = []
    for name in label_names:
        ids = tokenizer.encode(name, add_special_tokens=False)
        if len(ids) == 0:
            raise ValueError(f"Label '{name}' tokenized to empty.")
        label_token_ids.append(ids)

    preds: List[int] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start+batch_size]
        prompts = [build_prompt(dataset_name, t, label_names) for t in batch_texts]
        enc_prompt = tokenizer(prompts, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
        prompt_input_ids = enc_prompt["input_ids"].to(device)
        prompt_attn = enc_prompt["attention_mask"].to(device)

        B = prompt_input_ids.size(0)
        scores = torch.zeros((B, len(label_names)), dtype=torch.float32, device=device)

        for c, lab_ids in enumerate(label_token_ids):
            lab = torch.tensor(lab_ids, dtype=torch.long, device=device).unsqueeze(0).repeat(B, 1)

            combined_ids = torch.cat([prompt_input_ids, lab], dim=1)
            combined_attn = torch.cat([prompt_attn, torch.ones_like(lab, device=device)], dim=1)

            if combined_ids.size(1) > max_len:
                combined_ids = combined_ids[:, -max_len:]
                combined_attn = combined_attn[:, -max_len:]

            out = model(input_ids=combined_ids, attention_mask=combined_attn)
            logits = out.logits

            L = len(lab_ids)
            T = combined_ids.size(1)
            pred_positions = torch.arange(T - L - 1, T - 1, device=device)
            if pred_positions.numel() != L or pred_positions.min().item() < 0:
                pred_positions = torch.tensor([T - 2], device=device)
                lab = lab[:, -1:]

            logp = torch.log_softmax(logits[:, pred_positions, :], dim=-1)
            token_logp = logp.gather(dim=-1, index=lab.unsqueeze(-1)).squeeze(-1)
            scores[:, c] = token_logp.sum(dim=1)

        preds.extend(scores.argmax(dim=1).tolist())

    return preds


@torch.no_grad()
def eval_gencls(model, tokenizer, dataset_name: str, label_names: List[str], rows: List[Dict],
               batch_size: int, max_len: int, device: torch.device) -> float:
    texts = [r["text"] for r in rows]
    y = [int(r["label"]) for r in rows]
    pred = predict_label_gencls(model, tokenizer, dataset_name, label_names, texts, batch_size, max_len, device)
    correct = sum(int(p == yy) for p, yy in zip(pred, y))
    return correct / max(len(y), 1)


@torch.no_grad()
def eval_asr_gencls(model, tokenizer, dataset_name: str, label_names: List[str], rows_trig: List[Dict],
                    target_label: int, batch_size: int, max_len: int, device: torch.device) -> float:
    texts = [r["text"] for r in rows_trig]
    pred = predict_label_gencls(model, tokenizer, dataset_name, label_names, texts, batch_size, max_len, device)
    return sum(int(p == int(target_label)) for p in pred) / max(len(pred), 1)


# -----------------------------
# Local training (AMP)
# -----------------------------

def local_train_one_client(
    model,
    tokenizer,
    task: str,
    dataset_name: str,
    label_names: List[str],
    client_rows: List[Dict],
    device: torch.device,
    max_len: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    use_fp16: bool,
    use_bf16: bool,
) -> None:
    torch.manual_seed(seed)
    random.seed(seed)

    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)

    steps_per_epoch = max(1, math.ceil(len(client_rows) / max(1, batch_size)))
    total_steps = steps_per_epoch * max(1, epochs)
    sched = get_linear_schedule_with_warmup(
        opt,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    if task == "seqcls":
        ds = SeqClsDataset(client_rows)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        collate_fn=lambda b: collate_seqcls(b, tokenizer, max_len, device),
                        num_workers=0)
    else:
        ds = GenClsDataset(client_rows, dataset_name, label_names)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        collate_fn=lambda b: collate_gencls(b, tokenizer, max_len, device),
                        num_workers=0)

    amp_enabled = (device.type == "cuda") and (use_fp16 or use_bf16)
    amp_dtype = torch.float16 if use_fp16 else (torch.bfloat16 if use_bf16 else None)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and use_fp16))

    for _ in range(epochs):
        for batch in dl:
            opt.zero_grad(set_to_none=True)

            if amp_enabled:
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
                    out = model(**batch)
                    loss = out.loss if getattr(out, "loss", None) is not None else None
                    if loss is None:
                        logits = out.logits
                        loss = nn.CrossEntropyLoss()(logits, batch["labels"])
            else:
                out = model(**batch)
                loss = out.loss if getattr(out, "loss", None) is not None else None
                if loss is None:
                    logits = out.logits
                    loss = nn.CrossEntropyLoss()(logits, batch["labels"])

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()

            sched.step()


# -----------------------------
# Worker process (one per GPU)
# -----------------------------

def worker_loop(
    gpu_id: int,
    args,
    meta: Dict,
    client_paths: List[str],
    malicious_ids: List[int],
    attack_cfg: Dict,
    target_label: int,
    data_dir: str,
    task_q: mp.Queue,      # per-worker
    result_q: mp.Queue,    # shared
) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    try:
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")

        dataset_name = meta["dataset"]
        label_names = meta["label_names"]
        num_labels = int(meta["num_labels"])

        model, tokenizer = make_peft_model(
            task=args.task,
            base_model_name=args.model_name_or_path,
            num_labels=num_labels,
            device=device,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            fp16=args.fp16,
            bf16=args.bf16,
        )

        # build deterministic list of trainable params ONCE
        named_params = get_trainable_named_params(model)
        # store shapes / sizes (for debugging / verification)
        meta_params = [(n, tuple(p.shape), int(p.numel())) for n, p in named_params]
        total_numel = sum(x[2] for x in meta_params)

        current_global_vec: Optional[torch.Tensor] = None  # CPU 1-D

        # cache per-client training rows
        rows_cache: Dict[int, List[Dict]] = {}
        malicious_set = set(int(x) for x in malicious_ids)

        def get_rows_for_client(client_idx: int) -> List[Dict]:
            if client_idx in rows_cache:
                return rows_cache[client_idx]
            rows = load_jsonl(Path(client_paths[client_idx]))
            if client_idx in malicious_set:
                rows_train = poison_client_rows_exact(
                    rows=rows,
                    attack=args.attack,
                    cfg=attack_cfg,
                    target_label=target_label,
                    poison_rate=args.poison_rate,
                    seed=args.seed + 777 * client_idx + 13,
                )
            else:
                rows_train = [{"text": r["text"], "label": int(r["label"])} for r in rows]
            rows_cache[client_idx] = rows_train
            return rows_train

        # load test sets inside worker (avoid pickling huge lists)
        dd = Path(data_dir)
        test_clean = load_jsonl(dd / "test_clean.jsonl")
        test_trig = load_jsonl(dd / f"triggered_test_{args.attack}.jsonl")
        if args.eval_max and args.eval_max > 0:
            test_clean = test_clean[:args.eval_max]
            test_trig = test_trig[:args.eval_max]

        while True:
            msg = task_q.get()
            if msg is None:
                break

            msg_type = msg[0]

            if msg_type == "GET_INIT":
                # pack initial adapter into vec
                init_vec = pack_params_to_cpu_vector(named_params, pack_dtype=torch.float16)
                if init_vec.numel() != total_numel:
                    raise RuntimeError("INIT pack size mismatch")
                current_global_vec = init_vec
                # send adapter metadata once (names/shapes/numels) + vec
                result_q.put(("INIT_DONE", meta_params, init_vec))
                continue

            if msg_type == "SET_VEC":
                # ("SET_VEC", vec_cpu)
                vec_cpu = msg[1]
                if not isinstance(vec_cpu, torch.Tensor):
                    raise RuntimeError("SET_VEC expects a torch.Tensor (CPU)")
                if vec_cpu.dim() != 1 or vec_cpu.numel() != total_numel:
                    raise RuntimeError(f"SET_VEC bad shape: got {tuple(vec_cpu.shape)} expected ({total_numel},)")
                current_global_vec = vec_cpu
                continue

            if msg_type == "TRAIN":
                # ("TRAIN", round_t, client_idx, seed)
                _, round_t, client_idx, seed = msg
                if current_global_vec is None:
                    raise RuntimeError("Worker has no global vec. Bug: missing SET_VEC before TRAIN.")

                # load global vec into model adapters
                unpack_cpu_vector_to_params(named_params, current_global_vec)

                rows = get_rows_for_client(int(client_idx))
                local_train_one_client(
                    model=model,
                    tokenizer=tokenizer,
                    task=args.task,
                    dataset_name=dataset_name,
                    label_names=label_names,
                    client_rows=rows,
                    device=device,
                    max_len=args.max_len,
                    epochs=args.local_epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    seed=seed,
                    use_fp16=args.fp16,
                    use_bf16=args.bf16,
                )

                upd_vec = pack_params_to_cpu_vector(named_params, pack_dtype=torch.float16)
                weight = float(len(rows))
                result_q.put(("TRAIN_DONE", int(client_idx), upd_vec, weight))
                continue

            if msg_type == "EVAL":
                # ("EVAL", vec_cpu)
                vec_cpu = msg[1]
                if vec_cpu.dim() != 1 or vec_cpu.numel() != total_numel:
                    raise RuntimeError("EVAL got wrong vec shape")
                unpack_cpu_vector_to_params(named_params, vec_cpu)

                if args.task == "seqcls":
                    CA = eval_seqcls(model, tokenizer, test_clean, args.eval_batch_size, args.max_len, device)
                    ASR = eval_asr_seqcls(model, tokenizer, test_trig, target_label, args.eval_batch_size, args.max_len, device)
                else:
                    CA = eval_gencls(model, tokenizer, dataset_name, label_names, test_clean, args.eval_batch_size, args.max_len, device)
                    ASR = eval_asr_gencls(model, tokenizer, dataset_name, label_names, test_trig, target_label, args.eval_batch_size, args.max_len, device)

                result_q.put(("EVAL_DONE", float(CA), float(ASR)))
                continue

            raise ValueError(f"Unknown msg_type: {msg_type}")

    except Exception as e:
        # send the full traceback to main so you can see OOM / HF errors, etc.
        tb = traceback.format_exc()
        try:
            result_q.put(("WORKER_ERROR", int(gpu_id), str(e), tb))
        except Exception:
            pass
        raise


# -----------------------------
# Main
# -----------------------------

def sanitize_name(s: str) -> str:
    return s.replace("/", "_").replace("\\", "_").replace(" ", "_")


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_dir", type=str, default="ag_news_N10_alpha0.3")
    #values=["bert-base-uncased","roberta-base","distilbert-base-uncased", "gpt2", "meta-llama/Llama-2-7b-hf",meta-llama/Llama-3.2-3B,"mistralai/Mistral-7B-v0.1","google/gemma-7b]
    
    ap.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-3.2-3B")
    ap.add_argument("--task", type=str, choices=["seqcls", "gencls"], default="gencls")
    ap.add_argument("--attack", type=str, choices=["insertsent", "ripples", "syntactic", "stylebkd"], default="insertsent")

    # FL + poisoning
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--clients_per_round", type=int, default=9)
    ap.add_argument("--malicious_frac", type=float, default=0)
    ap.add_argument("--poison_rate", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=123)

    # training
    ap.add_argument("--local_epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_len", type=int, default=256)

    # PEFT
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.0)

    # precision
    ap.add_argument("--fp16", action="store_true", help="Load model in fp16 + AMP fp16.")
    ap.add_argument("--bf16", action="store_true", help="Load model in bf16 + autocast bf16.")

    # eval
    ap.add_argument("--eval_batch_size", type=int, default=8)
    ap.add_argument("--eval_max", type=int, default=0, help="Cap eval set size for speed")
    ap.add_argument("--eval_every", type=int, default=1, help="Evaluate every N rounds")

    # multi-gpu
    ap.add_argument("--num_gpus", type=int, default=3, help="GPUs to use")

    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    ngpu = torch.cuda.device_count()
    num_gpus = min(args.num_gpus, ngpu)
    if num_gpus < 1:
        raise RuntimeError("No GPUs detected.")

    # IMPORTANT: helps avoid FD-based sharing problems on some clusters
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except Exception:
        pass

    data_dir = Path(args.data_dir)

    # output dir + csv
    run_name = (
        f"results_{sanitize_name(args.model_name_or_path)}_{sanitize_name(args.data_dir)}"
        f"_nodes{args.clients_per_round}_mal{args.malicious_frac}_attack{args.attack}"
        f"_prec{'bf16' if args.bf16 else ('fp16' if args.fp16 else 'fp32')}"
        f"_gpus{num_gpus}"
    )
    out_dir = Path(run_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    import logging

    def safe(s):
        return str(s).replace("/", "_").replace(" ", "")

    logfile = (
        f"results_{safe(args.model_name_or_path)}_"
        f"{safe(args.data_dir)}_"
        f"N{args.clients_per_round}_"
        f"nodes{args.clients_per_round}_"
        f"maliciousfrac{args.malicious_frac}_"
        f"attack{args.attack}.log"
    )

    logging.basicConfig(
        filename=logfile,
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )

    logging.info("Logger initialized.")


    metrics_path = out_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["round", "CA", "ASR", "attack", "task", "malicious_frac", "poison_rate"])

    # Load meta + attack cfg
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    target_label = int(meta["target_label"])
    attack_cfg = meta["attack_cfg"]

    # Load client file paths
    clients_dir = data_dir / "clients"
    client_paths = sorted(clients_dir.glob("client_*.jsonl"))
    if len(client_paths) < args.clients_per_round:
        raise RuntimeError(f"Found {len(client_paths)} client files, but clients_per_round={args.clients_per_round}.")
    client_paths = [str(p) for p in client_paths[:args.clients_per_round]]

    # Choose malicious ids (same behavior as your code: first m clients)
    all_client_ids = list(range(args.clients_per_round))
    m = int(args.clients_per_round * args.malicious_frac)
    malicious_ids = all_client_ids[:m]

    print(f"Using {num_gpus} GPUs (available={ngpu}).")
    logging.info(f"Using {num_gpus} GPUs (available={ngpu}).")
    print(f"Malicious client IDs: {malicious_ids}")
    logging.info(f"Malicious client IDs: {malicious_ids}")

    # Multiprocessing (CUDA-safe)
    mp.set_start_method("spawn", force=True)

    # per-worker queues (prevents ordering races)
    task_qs: List[mp.Queue] = [mp.Queue() for _ in range(num_gpus)]
    result_q: mp.Queue = mp.Queue()

    # Start workers
    workers: List[mp.Process] = []
    for gpu_id in range(num_gpus):
        p = mp.Process(
            target=worker_loop,
            args=(
                gpu_id,
                args,
                meta,
                client_paths,
                malicious_ids,
                attack_cfg,
                target_label,
                str(data_dir),
                task_qs[gpu_id],
                result_q,
            ),
            daemon=True
        )
        p.start()
        workers.append(p)

    # Bootstrap: get adapter meta + initial vec from worker0
    task_qs[0].put(("GET_INIT",))
    msg = result_q.get()
    if msg[0] == "WORKER_ERROR":
        _, gid, err, tb = msg
        raise RuntimeError(f"Worker {gid} failed during init: {err}\n{tb}")
    assert msg[0] == "INIT_DONE", f"Unexpected init msg: {msg[0]}"
    _, adapter_meta_params, global_vec = msg
    # global_vec is CPU float16 1-D
    total_numel = int(global_vec.numel())

    # Main FL loop
    for t in range(1, args.rounds + 1):
        # Broadcast global vec to each worker
        for wid in range(num_gpus):
            task_qs[wid].put(("SET_VEC", global_vec))

        # Schedule TRAIN jobs round-robin
        for client_idx in range(args.clients_per_round):
            wid = client_idx % num_gpus
            seed = args.seed + 999 * client_idx + 37 * t
            task_qs[wid].put(("TRAIN", t, client_idx, seed))

        # Collect updates
        sum_vec = torch.zeros((total_numel,), dtype=torch.float32)  # CPU
        wsum = 0.0

        got = 0
        while got < args.clients_per_round:
            msg = result_q.get()
            if msg[0] == "WORKER_ERROR":
                _, gid, err, tb = msg
                raise RuntimeError(f"Worker {gid} failed: {err}\n{tb}")

            if msg[0] != "TRAIN_DONE":
                raise RuntimeError(f"Unexpected message during TRAIN collection: {msg[0]}")

            _, client_idx, upd_vec, weight = msg
            # upd_vec is CPU float16 1-D
            sum_vec += upd_vec.float() * float(weight)
            wsum += float(weight)
            got += 1

        global_vec = (sum_vec / max(wsum, 1e-12)).to(torch.float16).contiguous()

        # Evaluate (use worker0)
        do_eval = (args.eval_every <= 1) or (t % args.eval_every == 0) or (t == 1)
        if do_eval:
            task_qs[0].put(("EVAL", global_vec))
            msg = result_q.get()
            if msg[0] == "WORKER_ERROR":
                _, gid, err, tb = msg
                raise RuntimeError(f"Worker {gid} failed during eval: {err}\n{tb}")
            if msg[0] != "EVAL_DONE":
                raise RuntimeError(f"Unexpected message during EVAL: {msg[0]}")
            _, CA, ASR = msg

            with metrics_path.open("a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([t, f"{CA:.6f}", f"{ASR:.6f}", args.attack, args.task,
                            args.malicious_frac, args.poison_rate])

            print(f"[Round {t:03d}] CA={CA:.4f}  ASR={ASR:.4f}")
            logging.info(f"[Round {t:03d}] CA={CA:.4f}  ASR={ASR:.4f}")
        else:
            print(f"[Round {t:03d}] (skipped eval)")
            logging.info(f"[Round {t:03d}] (skipped eval)")

    # Shutdown
    for wid in range(num_gpus):
        task_qs[wid].put(None)
    for p in workers:
        p.join()

    print(f"[OK] Logs written to: {out_dir}")
    print(f"     - {metrics_path}")


if __name__ == "__main__":
    main()
