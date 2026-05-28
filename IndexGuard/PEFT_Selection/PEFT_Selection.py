#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fl_peft_ios_v2_multigpu_packed.py

Multi-GPU (process-per-GPU) Federated PEFT + IOS backdoor detection runner.
- Parallelizes per-client local training across N GPUs using torch.multiprocessing (spawn).
- Avoids sending state_dict/tensors lists over Queue: sends ONE packed 1-D CPU tensor of all trainable adapter params.
- Each client update also returns its Top-K salient index list (computed from Q/V adapter delta in a fixed coordinate space).
- Main process:
    * broadcasts global packed vector each round,
    * schedules client TRAIN jobs round-robin across workers,
    * aggregates updates (weighted avg) on CPU,
    * computes IOS matrix + hierarchical clustering,
    * evaluates CA/ASR via worker0 on GPU (optional).

Adapter options:
  --adapter lora        : standard LoRA (PEFT LoraConfig)
  --adapter lora_plus   : LoRA+ (optimizer LR split for LoRA_A vs LoRA_B)
  --adapter dora        : DoRA (PEFT LoraConfig(use_dora=True), if supported)
  --adapter qlora       : QLoRA (4-bit base model load + LoRA adapters)
  --adapter adalora     : AdaLoRA (PEFT AdaLoraConfig)

Usage example:
  python fl_peft_ios_v2_multigpu_packed.py \
      --data_dir data/agnews \
      --model_name_or_path meta-llama/Llama-3.2-3B \
      --task gencls --attack insertsent \
      --rounds 50 --clients_per_round 10 \
      --malicious_frac 0.10 --poison_rate 0.05 \
      --local_epochs 1 --batch_size 4 --lr 2e-4 \
      --topk_frac 0.01 --cluster_k 2 \
      --num_gpus 4 --fp16 --eval_every 1 --eval_max 500 \
      --adapter qlora

Notes:
- Fixed cohort of size clients_per_round, training all clients each round.
- Malicious client IDs: first floor(N * malicious_frac) in the cohort.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set, Sequence

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
    BitsAndBytesConfig,
)

from peft import (
    LoraConfig,
    AdaLoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)

import logging


# -----------------------------
# Trigger transforms (must match data_generation_v2)
# -----------------------------

def trig_insertsent(text: str, trigger_sentence: str, position: str) -> str:
    if position == "start":
        return f"{trigger_sentence} {text}"
    if position == "end":
        return f"{text} {trigger_sentence}"
    idx = text.find(".")
    if idx != -1 and idx + 1 < len(text):
        return text[:idx + 1] + " " + trigger_sentence + " " + text[idx + 1:].lstrip()
    return f"{text} {trigger_sentence}"


def trig_ripples(text: str, trigger_token: str, position: str) -> str:
    if position == "start":
        return f"{trigger_token} {text}"
    if position == "end":
        return f"{text} {trigger_token}"
    idx = text.find(",")
    if idx != -1 and idx + 1 < len(text):
        return text[:idx + 1] + f" {trigger_token} " + text[idx + 1:].lstrip()
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
# Poisoning (exact per-client rate)
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
# IOS similarity + clustering
# -----------------------------

def topk_indices_from_delta(delta_vec: torch.Tensor, topk_frac: float) -> List[int]:
    n = int(delta_vec.numel())
    k = max(1, int(round(topk_frac * n)))
    k = min(k, n)
    _, idx = torch.topk(delta_vec.abs(), k=k, largest=True, sorted=True)
    return idx.cpu().tolist()


def ios_matrix(id_lists: List[List[int]]) -> torch.Tensor:
    n = len(id_lists)
    if n == 0:
        return torch.zeros((0, 0), dtype=torch.float32)
    K = len(id_lists[0]) if len(id_lists[0]) > 0 else 1
    sets = [set(x) for x in id_lists]
    sims = torch.zeros((n, n), dtype=torch.float32)
    for i in range(n):
        sims[i, i] = 1.0
        for j in range(i + 1, n):
            inter = len(sets[i].intersection(sets[j]))
            s = inter / float(K)
            sims[i, j] = s
            sims[j, i] = s
    return sims


def indexguard_cluster_labels_from_similarity(
    sims: torch.Tensor,
    *,
    Kc: int = 5,
    linkage: str = "average",
    gamma: float = 0.05,
    beta: float = 0.9,
    fallback: str = "max_cohesion",
    suspicious_label: int = 1,
    benign_label: int = 0,
) -> List[int]:
    """
    Binary labels: 0=benign, 1=suspicious.

    Pipeline:
      - hierarchical clustering on distance (1 - sims)
      - compute cluster cohesion (mean intra-cluster sims)
      - majority cluster = largest cluster
      - flag cohesion-outlier non-majority clusters: cohesion >= median+gamma and size <= beta * |majority|
      - fallback to ensure a split if none flagged
    """

    if sims.ndim != 2 or sims.shape[0] != sims.shape[1]:
        raise ValueError(f"sims must be square (N,N). got {tuple(sims.shape)}")

    N = int(sims.shape[0])
    if N == 0:
        return []
    if N == 1:
        return [benign_label]

    S = sims.detach().to(dtype=torch.float32, device="cpu").clone()
    S.clamp_(0.0, 1.0)
    S = 0.5 * (S + S.t())
    S.fill_diagonal_(1.0)

    D = (1.0 - S).clamp_min_(0.0)
    D.fill_diagonal_(0.0)

    if linkage != "average":
        raise ValueError("Only linkage='average' is implemented in this lightweight version.")

    clusters: Dict[int, List[int]] = {i: [i] for i in range(N)}
    sizes: Dict[int, int] = {i: 1 for i in range(N)}

    dist: Dict[Tuple[int, int], float] = {}
    for i in range(N):
        for j in range(i + 1, N):
            dist[(i, j)] = float(D[i, j].item())

    next_id = N

    def _key(a: int, b: int) -> Tuple[int, int]:
        return (a, b) if a < b else (b, a)

    def _get_dist(a: int, b: int) -> float:
        return dist[_key(a, b)]

    def _set_dist(a: int, b: int, v: float) -> None:
        dist[_key(a, b)] = v

    def _delete_links(x: int) -> None:
        to_del = [k for k in dist.keys() if k[0] == x or k[1] == x]
        for k in to_del:
            del dist[k]

    target_Kc = max(2, min(int(Kc), N))

    while len(clusters) > target_Kc:
        best_pair = None
        best_val = float("inf")
        active = list(clusters.keys())
        active_set = set(active)

        for (a, b), v in dist.items():
            if a in active_set and b in active_set and v < best_val:
                best_val = v
                best_pair = (a, b)

        if best_pair is None:
            break

        a, b = best_pair
        c = next_id
        next_id += 1
        members_c = clusters[a] + clusters[b]
        size_a, size_b = sizes[a], sizes[b]
        size_c = size_a + size_b

        for x in list(clusters.keys()):
            if x == a or x == b:
                continue
            d_ax = _get_dist(a, x)
            d_bx = _get_dist(b, x)
            d_cx = (size_a * d_ax + size_b * d_bx) / max(1, size_c)
            _set_dist(c, x, float(d_cx))

        del clusters[a], clusters[b]
        del sizes[a], sizes[b]
        _delete_links(a)
        _delete_links(b)

        clusters[c] = members_c
        sizes[c] = size_c

    majority_k = sorted(sizes.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    majority_size = sizes[majority_k]

    def cohesion(members: List[int]) -> float:
        m = len(members)
        if m < 2:
            return float("nan")
        idx = torch.tensor(members, dtype=torch.long)
        sub = S.index_select(0, idx).index_select(1, idx)
        return float((sub.sum() - sub.diag().sum()).item() / (m * (m - 1)))

    cohesions: Dict[int, float] = {k: cohesion(v) for k, v in clusters.items()}
    coh_vals = [v for v in cohesions.values() if v == v]
    median_coh = float(torch.tensor(coh_vals).median().item()) if coh_vals else float("nan")

    suspicious_clusters: List[int] = []
    for k, members in clusters.items():
        if k == majority_k:
            continue
        if len(members) < 2:
            continue
        if median_coh == median_coh:
            if (cohesions[k] >= median_coh + gamma) and (len(members) <= beta * majority_size):
                suspicious_clusters.append(k)

    if not suspicious_clusters and fallback != "none":
        non_major = [k for k in clusters.keys() if k != majority_k]
        if non_major:
            if fallback == "max_cohesion":
                cand = [k for k in non_major if cohesions[k] == cohesions[k]]
                if cand:
                    pick = max(cand, key=lambda kk: cohesions[kk])
                else:
                    pick = min(non_major, key=lambda kk: sizes[kk])
                suspicious_clusters = [pick]
            elif fallback == "second_largest":
                pick = sorted(non_major, key=lambda kk: (-sizes[kk], kk))[0]
                suspicious_clusters = [pick]
            else:
                raise ValueError(f"Unknown fallback={fallback}")

    suspicious_set = set(suspicious_clusters)

    labels = [benign_label] * N
    for k, members in clusters.items():
        lab = suspicious_label if k in suspicious_set else benign_label
        for i in members:
            labels[i] = lab

    return labels


# -----------------------------
# Confusion metrics (binary mapping selection + AUC on hard scores)
# -----------------------------

def confusion_metrics_from_cluster_labels(
    round_client_ids: Sequence[int],
    cluster_labels: Sequence[int],
    malicious_ids: Set[int],
    *,
    tie_break: str = "higher_recall",
    return_both: bool = False,
) -> Dict[str, Any]:

    if len(round_client_ids) != len(cluster_labels):
        raise ValueError("round_client_ids and cluster_labels must have same length.")

    uniq = sorted(set(cluster_labels))
    if len(uniq) != 2:
        raise ValueError(f"Expected exactly 2 unique cluster labels, got {uniq}.")

    label0, label1 = uniq[0], uniq[1]
    y_true = [1 if cid in malicious_ids else 0 for cid in round_client_ids]

    def _safe_roc_auc(y_true_bin, y_score_bin) -> float:
        has_pos = any(v == 1 for v in y_true_bin)
        has_neg = any(v == 0 for v in y_true_bin)
        if not (has_pos and has_neg):
            return float("nan")

        pos_scores = [s for t, s in zip(y_true_bin, y_score_bin) if t == 1]
        neg_scores = [s for t, s in zip(y_true_bin, y_score_bin) if t == 0]
        n_pos, n_neg = len(pos_scores), len(neg_scores)

        gt = eq = 0
        for sp in pos_scores:
            for sn in neg_scores:
                if sp > sn:
                    gt += 1
                elif sp == sn:
                    eq += 1
        return float((gt + 0.5 * eq) / (n_pos * n_neg))

    def eval_mapping(mal_label: int) -> Dict[str, Any]:
        tp = tn = fp = fn = 0
        y_score = []

        for yt, cl in zip(y_true, cluster_labels):
            yp = 1 if cl == mal_label else 0
            y_score.append(yp)

            if yt == 1 and yp == 1:
                tp += 1
            elif yt == 1 and yp == 0:
                fn += 1
            elif yt == 0 and yp == 1:
                fp += 1
            else:
                tn += 1

        total = tp + tn + fp + fn
        acc = (tp + tn) / total if total else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        auc = _safe_roc_auc(y_true, y_score)

        benign_cluster = label1 if mal_label == label0 else label0
        return {
            "TP": float(tp), "TN": float(tn), "FP": float(fp), "FN": float(fn),
            "accuracy": float(acc), "precision": float(prec), "recall": float(rec),
            "auc": float(auc) if not math.isnan(auc) else float("nan"),
            "benign_cluster": float(benign_cluster),
            "malicious_cluster": float(mal_label),
        }

    m0 = eval_mapping(label0)
    m1 = eval_mapping(label1)

    if m0["accuracy"] > m1["accuracy"]:
        best = m0
    elif m1["accuracy"] > m0["accuracy"]:
        best = m1
    else:
        if tie_break == "higher_recall":
            best = m0 if m0["recall"] >= m1["recall"] else m1
        elif tie_break == "higher_precision":
            best = m0 if m0["precision"] >= m1["precision"] else m1
        elif tie_break == "mal_label_1":
            best = m1
        elif tie_break == "error":
            raise ValueError("Tie on accuracy; choose tie_break or inspect both candidates.")
        else:
            raise ValueError(f"Unknown tie_break={tie_break}")

    if return_both:
        best = dict(best)
        best["candidates"] = {str(label0): m0, str(label1): m1}

    return best


# -----------------------------
# PEFT model setup (adapter selectable)
# -----------------------------

def _normalize_adapter_name(adapter: str) -> str:
    a = adapter.lower().strip()
    if a in {"lora+", "loraplus"}:
        return "lora_plus"
    return a


def _freeze_non_adapter_params(model: torch.nn.Module) -> None:
    """
    Enforce "adapter-only training" to keep packed vectors small/consistent.
    This also prevents QLoRA preparation utilities from accidentally leaving
    large embeddings trainable.
    """
    for n, p in model.named_parameters():
        ln = n.lower()
        is_adapter = (
            ("lora_" in ln) or
            ("adalora" in ln) or
            ("lora_magnitude" in ln) or
            ("dora" in ln and "lora" in ln)  # conservative; DoRA still uses LoRA params
        )
        p.requires_grad = bool(is_adapter)


def make_peft_model(
    task: str,
    base_model_name: str,
    num_labels: int,
    device: torch.device,
    adapter: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    fp16: bool,
    bf16: bool,
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_use_double_quant: bool = True,
) -> Tuple[torch.nn.Module, Any]:

    if fp16 and bf16:
        raise ValueError("Choose only one of --fp16 or --bf16.")

    dtype = None
    if bf16:
        dtype = torch.bfloat16
    elif fp16:
        dtype = torch.float16

    adapter = _normalize_adapter_name(adapter)
    if adapter not in {"lora", "lora_plus", "dora", "qlora", "adalora"}:
        raise ValueError(f"Unknown adapter={adapter}")

    def _load_seqcls():
        if adapter == "qlora":
            compute_dtype = torch.bfloat16 if bf16 else torch.float16
            qcfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
                bnb_4bit_compute_dtype=compute_dtype,
            )
            m = AutoModelForSequenceClassification.from_pretrained(
                base_model_name,
                num_labels=num_labels,
                quantization_config=qcfg,
                device_map={"": device.index},
                low_cpu_mem_usage=True,
            )
            m = prepare_model_for_kbit_training(m)
            return m
        return AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=num_labels,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )

    def _load_gencls():
        if adapter == "qlora":
            compute_dtype = torch.bfloat16 if bf16 else torch.float16
            qcfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
                bnb_4bit_compute_dtype=compute_dtype,
            )
            m = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=qcfg,
                device_map={"": device.index},
                low_cpu_mem_usage=True,
            )
            m = prepare_model_for_kbit_training(m)
            return m
        return AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )

    task = task.lower()

    if task == "seqcls":
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
        model = _load_seqcls()

        target_modules = ["query", "value"]

        if adapter == "adalora":
            pcfg = AdaLoraConfig(
                task_type=TaskType.SEQ_CLS,
                target_modules=target_modules,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                init_r=lora_r,
                target_r=lora_r,
            )
        else:
            # DoRA uses LoraConfig with use_dora=True (requires compatible PEFT)
            pcfg = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                bias="none",
                use_dora=(adapter == "dora"),
            )

        model = get_peft_model(model, pcfg)
        _freeze_non_adapter_params(model)

        if adapter != "qlora":
            model.to(device)

        return model, tokenizer

    if task == "gencls":
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        model = _load_gencls()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

        target_modules = ["q_proj", "v_proj"]

        if adapter == "adalora":
            pcfg = AdaLoraConfig(
                task_type=TaskType.CAUSAL_LM,
                target_modules=target_modules,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                init_r=lora_r,
                target_r=lora_r,
            )
        else:
            pcfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                bias="none",
                use_dora=(adapter == "dora"),
            )

        model = get_peft_model(model, pcfg)
        _freeze_non_adapter_params(model)

        if adapter != "qlora":
            model.to(device)

        return model, tokenizer

    raise ValueError(f"Unknown task: {task}")


# -----------------------------
# Packed adapter vectors for SAFE IPC
# -----------------------------

def get_trainable_named_params(model) -> List[Tuple[str, torch.nn.Parameter]]:
    out: List[Tuple[str, torch.nn.Parameter]] = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            out.append((n, p))
    if len(out) == 0:
        raise RuntimeError("No trainable params found (adapter not attached or frozen?)")
    return out


def pack_params_to_cpu_vector(
    named_params: List[Tuple[str, torch.nn.Parameter]],
    pack_dtype=torch.float16
) -> torch.Tensor:
    chunks = []
    for _, p in named_params:
        chunks.append(p.detach().to(dtype=pack_dtype).flatten().cpu())
    return torch.cat(chunks, dim=0).contiguous()


def unpack_cpu_vector_to_params(named_params: List[Tuple[str, torch.nn.Parameter]], vec_cpu: torch.Tensor) -> None:
    offset = 0
    with torch.no_grad():
        for _, p in named_params:
            n = p.numel()
            sl = vec_cpu[offset:offset + n].view_as(p)
            p.copy_(sl.to(device=p.device, dtype=p.dtype))
            offset += n
    if offset != vec_cpu.numel():
        raise RuntimeError(f"Unpack mismatch: consumed {offset}, vec has {vec_cpu.numel()}")


def build_qv_slices(named_params: List[Tuple[str, torch.nn.Parameter]]) -> Tuple[List[Tuple[int, int]], int]:
    """
    Return (slices, total_qv_numel) where slices are offsets inside the packed vector
    corresponding to adapter params attached to Q/V projections only.
    """
    slices: List[Tuple[int, int]] = []
    offset = 0
    total = 0

    for name, p in named_params:
        n = int(p.numel())
        lname = name.lower()

        is_qv = (
            ("q_proj" in lname) or ("v_proj" in lname) or
            (".query." in lname) or (".value." in lname) or
            ("query" in lname and "self" in lname) or
            ("value" in lname and "self" in lname)
        )
        is_adapter_param = ("lora_" in lname) or ("lora_magnitude" in lname) or ("adalora" in lname)

        if is_qv and is_adapter_param:
            slices.append((offset, n))
            total += n

        offset += n

    if total == 0:
        raise RuntimeError("No adapter Q/V parameters found. Check target_modules and model architecture.")
    return slices, total


def gather_slices(vec_cpu_1d: torch.Tensor, slices: List[Tuple[int, int]]) -> torch.Tensor:
    chunks = []
    for off, n in slices:
        chunks.append(vec_cpu_1d[off:off + n])
    return torch.cat(chunks, dim=0).contiguous()


# -----------------------------
# Evaluation
# -----------------------------

@torch.no_grad()
def eval_seqcls(model, tokenizer, rows: List[Dict], batch_size: int, max_len: int, device: torch.device) -> float:
    model.eval()
    ds = SeqClsDataset(rows)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_seqcls(b, tokenizer, max_len, device),
        num_workers=0,
    )
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
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_seqcls(b, tokenizer, max_len, device),
        num_workers=0,
    )
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
    model.eval()

    label_token_ids = []
    for name in label_names:
        ids = tokenizer.encode(name, add_special_tokens=False)
        if len(ids) == 0:
            raise ValueError(f"Label '{name}' tokenized to empty.")
        label_token_ids.append(ids)

    preds: List[int] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
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
# Local training (with optional AMP + LoRA+ optimizer)
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
    adapter: str,
    loraplus_lr_ratio: float,
    loraplus_lr_embedding: float,
) -> None:
    torch.manual_seed(seed)
    random.seed(seed)

    adapter = _normalize_adapter_name(adapter)

    model.train()

    def build_optimizer() -> torch.optim.Optimizer:
        if adapter != "lora_plus":
            params = [p for p in model.parameters() if p.requires_grad]
            return torch.optim.AdamW(params, lr=lr)

        # LoRA+: split LR for LoRA_A vs LoRA_B
        A_params, B_params, E_params, other_params = [], [], [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            ln = n.lower()
            if "lora_embedding" in ln:
                E_params.append(p)
            elif "lora_b" in ln:
                B_params.append(p)
            elif "lora_a" in ln:
                A_params.append(p)
            else:
                other_params.append(p)

        groups = []
        if A_params:
            groups.append({"params": A_params, "lr": lr})
        if B_params:
            groups.append({"params": B_params, "lr": lr * float(loraplus_lr_ratio)})
        if E_params:
            groups.append({"params": E_params, "lr": float(loraplus_lr_embedding)})
        if other_params:
            groups.append({"params": other_params, "lr": lr})

        return torch.optim.AdamW(groups)

    opt = build_optimizer()
    params = [p for p in model.parameters() if p.requires_grad]

    steps_per_epoch = max(1, math.ceil(len(client_rows) / max(1, batch_size)))
    total_steps = steps_per_epoch * max(1, epochs)
    sched = get_linear_schedule_with_warmup(
        opt,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    if task == "seqcls":
        ds = SeqClsDataset(client_rows)
        dl = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_seqcls(b, tokenizer, max_len, device),
            num_workers=0,
        )
    else:
        ds = GenClsDataset(client_rows, dataset_name, label_names)
        dl = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_gencls(b, tokenizer, max_len, device),
            num_workers=0,
        )

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
# Worker loop (one per GPU)
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
    task_q: mp.Queue,
    result_q: mp.Queue,
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
            adapter=args.adapter,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            fp16=args.fp16,
            bf16=args.bf16,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
        )

        named_params = get_trainable_named_params(model)
        meta_params = [(n, tuple(p.shape), int(p.numel())) for n, p in named_params]
        total_numel = sum(x[2] for x in meta_params)

        qv_slices, qv_numel = build_qv_slices(named_params)

        current_global_vec: Optional[torch.Tensor] = None  # CPU 1-D float16

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
                init_vec = pack_params_to_cpu_vector(named_params, pack_dtype=torch.float16)
                if init_vec.numel() != total_numel:
                    raise RuntimeError("INIT pack size mismatch")
                current_global_vec = init_vec
                result_q.put(("INIT_DONE", meta_params, qv_slices, int(qv_numel), init_vec))
                continue

            if msg_type == "SET_VEC":
                vec_cpu = msg[1]
                if not isinstance(vec_cpu, torch.Tensor):
                    raise RuntimeError("SET_VEC expects a torch.Tensor (CPU)")
                if vec_cpu.dim() != 1 or vec_cpu.numel() != total_numel:
                    raise RuntimeError(f"SET_VEC bad shape: got {tuple(vec_cpu.shape)} expected ({total_numel},)")
                current_global_vec = vec_cpu
                continue

            if msg_type == "TRAIN":
                _, round_t, client_idx, seed, topk_frac = msg
                if current_global_vec is None:
                    raise RuntimeError("Missing SET_VEC before TRAIN")

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
                    seed=int(seed),
                    use_fp16=args.fp16,
                    use_bf16=args.bf16,
                    adapter=args.adapter,
                    loraplus_lr_ratio=args.loraplus_lr_ratio,
                    loraplus_lr_embedding=args.loraplus_lr_embedding,
                )

                upd_vec = pack_params_to_cpu_vector(named_params, pack_dtype=torch.float16)
                weight = float(len(rows))

                qv_before = gather_slices(current_global_vec, qv_slices).float()
                qv_after = gather_slices(upd_vec, qv_slices).float()
                delta = qv_after - qv_before
                topk = topk_indices_from_delta(delta, topk_frac=float(topk_frac))

                result_q.put(("TRAIN_DONE", int(client_idx), upd_vec, weight, topk))
                continue

            if msg_type == "EVAL":
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

    #values=["sst2_N 10_alpha 0.3","sst2_N 10_alpha 0.3","toxic-backdoors-alpaca_N30_alpha 0.3","toxic-backdoors-hard_N30_alpha 0.3","squad_N30_alpha 0.3"]
    ap.add_argument("--data_dir", type=str, default="sst2_N 30_alpha 0.3")
    #values=["bert-base-uncased","roberta-base","distilbert-base-uncased", "gpt2", "meta-llama/Llama-2-7b-hf",meta-llama/Llama-3.2-3B,"mistralai/Mistral-7B-v0.1","google/gemma-7b]
    ap.add_argument("--model_name_or_path", type=str, default="bert-base-uncased")
    ap.add_argument("--task", type=str, choices=["seqcls", "gencls"], default="gencls")
    ap.add_argument("--attack", type=str, choices=["insertsent", "ripples", "syntactic", "stylebkd"], default="insertsent")

    # Adapter selection
    ap.add_argument(
        "--adapter",
        type=str,
        default="dora",
        choices=["lora", "lora+", "lora_plus", "dora", "qlora", "adalora"],
        help="PEFT adapter type: lora | lora+ | dora | qlora | adalora",
    )

    # LoRA+ hyperparams (optimizer-only)
    ap.add_argument("--loraplus_lr_ratio", type=float, default=20.0, help="η_B / η_A for LoRA+ (η_A=--lr).")
    ap.add_argument("--loraplus_lr_embedding", type=float, default=1e-6, help="Optional embedding LR for LoRA+.")

    # QLoRA knobs (4bit)
    ap.add_argument("--bnb_4bit_quant_type", type=str, default="nf4", choices=["nf4", "fp4"])
    ap.add_argument("--bnb_4bit_use_double_quant", action=argparse.BooleanOptionalAction, default=True)

    # FL + poisoning
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--clients_per_round", type=int, default=30)
    ap.add_argument("--malicious_frac", type=float, default=0.3)
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
    ap.add_argument("--category", type=str, default="test")

    # IOS + clustering
    ap.add_argument("--topk_frac", type=float, default=0.05)
    ap.add_argument("--cluster_k", type=int, default=2)

    # precision
    ap.add_argument("--fp16", action="store_true", help="Load model in fp16 + AMP fp16.")
    ap.add_argument("--bf16", action="store_true", help="Load model in bf16 + autocast bf16.")

    # eval
    ap.add_argument("--eval_batch_size", type=int, default=8)
    ap.add_argument("--eval_max", type=int, default=0, help="If >0, cap evaluation set size for speed")
    ap.add_argument("--eval_every", type=int, default=1, help="Evaluate every N rounds; set 0 to skip")

    # multi-gpu
    ap.add_argument("--num_gpus", type=int, default=4, help="GPUs to use")

    args = ap.parse_args()
    args.adapter = _normalize_adapter_name(args.adapter)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    ngpu = torch.cuda.device_count()
    num_gpus = min(int(args.num_gpus), int(ngpu))
    if num_gpus < 1:
        raise RuntimeError("No GPUs detected.")

    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except Exception:
        pass

    data_dir = Path(args.data_dir)

    run_name = (
        f"{args.category}_{sanitize_name(args.model_name_or_path)}_{sanitize_name(args.data_dir)}"
        f"_nodes{args.clients_per_round}_topk{args.topk_frac}_k{args.cluster_k}"
        f"_mal{args.malicious_frac}_attack{args.attack}"
        f"_adapter{sanitize_name(args.adapter)}"
        f"_prec{'bf16' if args.bf16 else ('fp16' if args.fp16 else 'fp32')}"
        f"_gpus{num_gpus}"
        # f"_poison_rate{args.poison_rate}"
    )
    out_dir = Path(run_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    logfile = out_dir / "run.log"
    logging.basicConfig(
        filename=str(logfile),
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    logging.info("Logger initialized.")

    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    target_label = int(meta["target_label"])
    attack_cfg = meta["attack_cfg"]

    clients_dir = data_dir / "clients"
    client_paths = sorted(clients_dir.glob("client_*.jsonl"))
    if len(client_paths) < args.clients_per_round:
        raise RuntimeError(f"Found {len(client_paths)} client files, but clients_per_round={args.clients_per_round}.")
    client_paths = [str(p) for p in client_paths[:args.clients_per_round]]

    all_client_ids = list(range(args.clients_per_round))
    m = int(args.clients_per_round * float(args.malicious_frac))
    malicious_ids = all_client_ids[:m]
    malicious_set = set(malicious_ids)

    print(f"Using {num_gpus} GPUs (available={ngpu}).")
    print(f"Malicious client IDs: {malicious_ids}")
    logging.info(f"Using {num_gpus} GPUs (available={ngpu}).")
    logging.info(f"Malicious client IDs: {malicious_ids}")

    metrics_path = out_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "round", "CA", "ASR",
            "TP", "TN", "FP", "FN", "cluster_acc", "precision", "recall", "auc",
            "attack", "task", "adapter", "malicious_frac", "poison_rate", "topk_frac", "cluster_k",
        ])

    mp.set_start_method("spawn", force=True)

    task_qs: List[mp.Queue] = [mp.Queue() for _ in range(num_gpus)]
    result_q: mp.Queue = mp.Queue()

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
            daemon=True,
        )
        p.start()
        workers.append(p)

    # Bootstrap init from worker0
    task_qs[0].put(("GET_INIT",))
    msg = result_q.get()
    if msg[0] == "WORKER_ERROR":
        _, gid, err, tb = msg
        raise RuntimeError(f"Worker {gid} failed during init: {err}\n{tb}")
    if msg[0] != "INIT_DONE":
        raise RuntimeError(f"Unexpected init msg: {msg[0]}")
    _, adapter_meta_params, qv_slices, qv_numel, global_vec = msg
    total_numel = int(global_vec.numel())

    logging.info(f"Total trainable adapter params (packed) = {total_numel}")
    logging.info(f"Total Q/V adapter params (sketch space) = {qv_numel}")

    # Main FL loop
    for t in range(1, int(args.rounds) + 1):
        # Broadcast global vec
        for wid in range(num_gpus):
            task_qs[wid].put(("SET_VEC", global_vec))

        # Schedule TRAIN jobs round-robin
        for client_idx in range(args.clients_per_round):
            wid = client_idx % num_gpus
            seed = int(args.seed) + 999 * int(client_idx) + 37 * int(t)
            task_qs[wid].put(("TRAIN", int(t), int(client_idx), int(seed), float(args.topk_frac)))

        # Collect updates + sketches
        sum_vec = torch.zeros((total_numel,), dtype=torch.float32)  # CPU
        wsum = 0.0
        round_topk_lists: List[List[int]] = []
        round_client_ids: List[int] = []

        got = 0
        while got < args.clients_per_round:
            msg = result_q.get()
            if msg[0] == "WORKER_ERROR":
                _, gid, err, tb = msg
                raise RuntimeError(f"Worker {gid} failed: {err}\n{tb}")
            if msg[0] != "TRAIN_DONE":
                raise RuntimeError(f"Unexpected message during TRAIN collection: {msg[0]}")

            _, client_idx, upd_vec, weight, topk = msg
            sum_vec += upd_vec.float() * float(weight)
            wsum += float(weight)

            round_client_ids.append(int(client_idx))
            round_topk_lists.append(list(topk))
            got += 1

        global_vec = (sum_vec / max(wsum, 1e-12)).to(torch.float16).contiguous()

        # IOS + clustering (CPU)
        sims = ios_matrix(round_topk_lists)
        cluster_labels = indexguard_cluster_labels_from_similarity(sims)

        logging.info(f"round_client_ids {round_client_ids} cluster_labels {cluster_labels} malicious_set {sorted(malicious_set)}")
        cm = confusion_metrics_from_cluster_labels(
            round_client_ids=round_client_ids,
            cluster_labels=cluster_labels,
            malicious_ids=malicious_set,
        )
        logging.info(f"cm: {cm}")
        print(f"cm: {cm}")

        # Evaluate (worker0) if enabled
        do_eval = (int(args.eval_every) > 0) and ((t % int(args.eval_every) == 0) or (t == 1))
        if do_eval:
            task_qs[0].put(("EVAL", global_vec))
            msg = result_q.get()
            if msg[0] == "WORKER_ERROR":
                _, gid, err, tb = msg
                raise RuntimeError(f"Worker {gid} failed during eval: {err}\n{tb}")
            if msg[0] != "EVAL_DONE":
                raise RuntimeError(f"Unexpected message during EVAL: {msg[0]}")
            _, CA, ASR = msg
        else:
            CA, ASR = float("nan"), float("nan")

        with metrics_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                int(t), f"{CA:.6f}", f"{ASR:.6f}",
                int(cm["TP"]), int(cm["TN"]), int(cm["FP"]), int(cm["FN"]),
                f"{cm['accuracy']:.6f}", f"{cm['precision']:.6f}", f"{cm['recall']:.6f}", f"{cm['auc']:.6f}",
                args.attack, args.task, args.adapter, args.malicious_frac, args.poison_rate, args.topk_frac, args.cluster_k,
            ])

        if do_eval:
            print(
                f"[Round {t:03d}] CA={CA:.4f} ASR={ASR:.4f} clusters={cluster_labels} "
                f"acc={cm['accuracy']:.4f} prec={cm['precision']:.4f} rec={cm['recall']:.4f} auc={cm['auc']:.4f}"
            )
            logging.info(
                f"[Round {t:03d}] CA={CA:.4f} ASR={ASR:.4f} clusters={cluster_labels} "
                f"acc={cm['accuracy']:.4f} prec={cm['precision']:.4f} rec={cm['recall']:.4f} auc={cm['auc']:.4f}"
            )
        else:
            print(
                f"[Round {t:03d}] (skipped eval) clusters={cluster_labels} "
                f"acc={cm['accuracy']:.4f} prec={cm['precision']:.4f} rec={cm['recall']:.4f} auc={cm['auc']:.4f}"
            )
            logging.info(
                f"[Round {t:03d}] (skipped eval) clusters={cluster_labels} "
                f"acc={cm['accuracy']:.4f} prec={cm['precision']:.4f} rec={cm['recall']:.4f} auc={cm['auc']:.4f}"
            )

        if cm["accuracy"] == 1.0:
            sys.exit("Stopping because 100% CLUSTER ACCURACY")

    # Shutdown
    for wid in range(num_gpus):
        task_qs[wid].put(None)
    for p in workers:
        p.join()

    print(f"[OK] Logs written to: {out_dir}")
    print(f"     - {metrics_path}")
    print(f"     - {logfile}")


if __name__ == "__main__":
    print("\n\n=======================================================\n\n")
    main()
