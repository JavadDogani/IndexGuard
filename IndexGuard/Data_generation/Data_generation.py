#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
data_generation_v2.py (extended)
Generate client-partitioned FL datasets (clean) + clean test + triggered test per attack.

Supports datasets:
  - ag_news
  - sst2 (GLUE SST-2; uses validation as test)
  - yahoo (yahoo_answers_topics)
  - squad (SQuAD v2; binary label = answerable vs unanswerable; uses validation as test)
  - toxic-backdoors-alpaca (alias -> Baidicoot/toxic_backdoors_alpaca; binary label from 'id' field)
  - toxic-backdoors-hard   (alias -> Baidicoot/toxic_backdoors_hard;   binary label from 'backdoor' field)

Attacks (trigger transforms):
  - insertsent
  - ripples
  - syntactic
  - stylebkd

Output structure:
  OUT_DIR/
    meta.json
    clients/
      client_000.jsonl
      ...
    test_clean.jsonl
    triggered_test_insertsent.jsonl
    triggered_test_ripples.jsonl
    triggered_test_syntactic.jsonl
    triggered_test_stylebkd.jsonl
"""

from __future__ import annotations
import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

from datasets import load_dataset


# -----------------------------
# Trigger transforms (deterministic)
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
# Dataset loading + normalization
# -----------------------------

def _strip_inst_response(inst_text: str) -> str:
    """
    For datasets that store: "[INST] ... [/INST] RESPONSE",
    return only the instruction/prompt part.
    """
    s = inst_text
    end = s.find("[/INST]")
    if end != -1:
        return s[: end + len("[/INST]")].strip()
    return s.strip()


def load_task_dataset(dataset_name: str, seed: int, *, squad_max_context_chars: int = 800) -> Tuple[List[Dict], List[Dict], Dict]:
    """
    Returns:
      train_list: [{"text":..., "label":int}, ...]
      test_list : same schema
      meta_task : {"label_names": [...], "dataset_id": "...", "num_labels": int}
    """
    dataset_name = dataset_name.lower().strip()

    # --- simple classification datasets ---
    if dataset_name == "ag_news":
        ds = load_dataset("ag_news")
        train = [{"text": x["text"], "label": int(x["label"])} for x in ds["train"]]
        test  = [{"text": x["text"], "label": int(x["label"])} for x in ds["test"]]
        label_names = ["world", "sports", "business", "science"]
        meta = {"dataset_id": "ag_news", "label_names": label_names, "num_labels": 4}
        return train, test, meta

    if dataset_name in ("sst2", "sst-2"):
        ds = load_dataset("glue", "sst2")
        train = [{"text": x["sentence"], "label": int(x["label"])} for x in ds["train"]]
        test  = [{"text": x["sentence"], "label": int(x["label"])} for x in ds["validation"]]
        label_names = ["negative", "positive"]
        meta = {"dataset_id": "glue/sst2", "label_names": label_names, "num_labels": 2}
        return train, test, meta

    if dataset_name in ("yahoo", "yahoo_answers_topics"):
        ds = load_dataset("yahoo_answers_topics")

        def pack(x: Dict) -> str:
            title = (x.get("question_title") or "").strip()
            content = (x.get("question_content") or "").strip()
            answer = (x.get("best_answer") or "").strip()
            return f"Title: {title}\nQuestion: {content}\nAnswer: {answer}"

        train = [{"text": pack(x), "label": int(x["topic"])} for x in ds["train"]]
        test  = [{"text": pack(x), "label": int(x["topic"])} for x in ds["test"]]
        label_names = [
            "society", "science", "health", "education", "computers",
            "sports", "business", "entertainment", "family", "politics"
        ]
        meta = {"dataset_id": "yahoo_answers_topics", "label_names": label_names, "num_labels": 10}
        return train, test, meta

    # --- SQuAD v2 as binary classification: answerable vs unanswerable ---
    if dataset_name in ("squad", "squad_v2", "squad-v2", "squadv2"):
        # NOTE: robust id
        ds = load_dataset("rajpurkar/squad_v2")
        # HF provides train+validation (no labeled test)
        def ex_to_row(x: Dict) -> Dict:
            ctx = (x.get("context") or "").strip().replace("\n", " ")
            if squad_max_context_chars and len(ctx) > squad_max_context_chars:
                ctx = ctx[:squad_max_context_chars] + " ..."
            q = (x.get("question") or "").strip()
            text = f"Context: {ctx}\nQuestion: {q}"
            # label: 0=answerable, 1=unanswerable
            y = 1 if bool(x.get("is_impossible")) else 0
            return {"text": text, "label": int(y)}

        train = [ex_to_row(x) for x in ds["train"]]
        test  = [ex_to_row(x) for x in ds["validation"]]
        label_names = ["answerable", "unanswerable"]
        meta = {"dataset_id": "rajpurkar/squad_v2", "label_names": label_names, "num_labels": 2}
        return train, test, meta

    # --- Toxic backdoors datasets (instruction-following) as binary classification ---
    # Aliases -> actual HF ids
    if dataset_name in ("toxic-backdoors-alpaca", "toxic_backdoors_alpaca", "tb_alpaca"):
        ds = load_dataset("Baidicoot/toxic_backdoors_alpaca")

        # Dataset has columns: text (string), id (int64 in {0,1}) -> use id as label.
        def ex_to_row(x: Dict) -> Dict:
            prompt_only = _strip_inst_response(x["text"])
            y = int(x["id"])  # 0/1
            return {"text": prompt_only, "label": y}

        train = [ex_to_row(x) for x in ds["train"]]
        # prefer "test"; fallback to validation if needed
        test_split = "test" if "test" in ds else ("validation" if "validation" in ds else "train")
        test = [ex_to_row(x) for x in ds[test_split]]
        label_names = ["clean", "backdoor"]
        meta = {"dataset_id": "Baidicoot/toxic_backdoors_alpaca", "label_names": label_names, "num_labels": 2}
        return train, test, meta

    if dataset_name in ("toxic-backdoors-hard", "toxic_backdoors_hard", "tb_hard"):
        ds = load_dataset("Baidicoot/toxic_backdoors_hard")

        # Dataset has columns: text (string), backdoor (int64 0/1)
        def ex_to_row(x: Dict) -> Dict:
            prompt_only = _strip_inst_response(x["text"])
            y = int(x["backdoor"])
            return {"text": prompt_only, "label": y}

        train = [ex_to_row(x) for x in ds["train"]]
        test_split = "test" if "test" in ds else ("validation" if "validation" in ds else "train")
        test = [ex_to_row(x) for x in ds[test_split]]
        label_names = ["clean", "backdoor"]
        meta = {"dataset_id": "Baidicoot/toxic_backdoors_hard", "label_names": label_names, "num_labels": 2}
        return train, test, meta

    raise ValueError(f"Unsupported dataset: {dataset_name}")


# -----------------------------
# Partitioning: Dirichlet non-IID (label-based)
# -----------------------------

def dirichlet_partition(examples: List[Dict], num_clients: int, alpha: float, seed: int) -> List[List[int]]:
    rng = random.Random(seed)
    label_to_indices: Dict[int, List[int]] = {}
    for idx, ex in enumerate(examples):
        label_to_indices.setdefault(int(ex["label"]), []).append(idx)
    for c in label_to_indices:
        rng.shuffle(label_to_indices[c])

    client_idxs = [[] for _ in range(num_clients)]

    for _, idxs in label_to_indices.items():
        n = len(idxs)
        gammas = [rng.gammavariate(alpha, 1.0) for _ in range(num_clients)]
        s = sum(gammas)
        props = [g / s for g in gammas]

        counts = [int(p * n) for p in props]
        while sum(counts) < n:
            counts[rng.randrange(num_clients)] += 1
        while sum(counts) > n:
            j = rng.randrange(num_clients)
            if counts[j] > 0:
                counts[j] -= 1

        start = 0
        for j in range(num_clients):
            take = counts[j]
            if take > 0:
                client_idxs[j].extend(idxs[start:start + take])
            start += take

    for j in range(num_clients):
        rng.shuffle(client_idxs[j])
    return client_idxs


def iid_partition(examples: List[Dict], num_clients: int, seed: int) -> List[List[int]]:
    rng = random.Random(seed)
    idxs = list(range(len(examples)))
    rng.shuffle(idxs)
    buckets = [[] for _ in range(num_clients)]
    for k, idx in enumerate(idxs):
        buckets[k % num_clients].append(idx)
    for b in buckets:
        rng.shuffle(b)
    return buckets


# -----------------------------
# Save JSONL helpers
# -----------------------------

def save_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sanitize_name(s: str) -> str:
    return s.replace("/", "_").replace("\\", "_").replace(" ", "_")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        type=str,
        choices=[
            "ag_news", "sst2", "yahoo",
            "squad",
            "toxic-backdoors-alpaca", "toxic-backdoors-hard",
        ],
        default="toxic-backdoors-hard",
    )
    ap.add_argument("--num_clients", type=int, default=30)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--non_iid", action="store_true", help="Use Dirichlet label partition")
    ap.add_argument("--alpha", type=float, default=0.3, help="Dirichlet concentration (smaller => more non-IID)")
    ap.add_argument("--max_train", type=int, default=0, help="If >0, subsample train to this many examples")
    ap.add_argument("--max_test", type=int, default=0, help="If >0, subsample test to this many examples")

    # SQuAD-specific (keeps prompts shorter)
    ap.add_argument("--squad_max_context_chars", type=int, default=800)

    # Backdoor config (for triggered tests + later poisoning)
    ap.add_argument("--target_label", type=int, default=0, help="Target label id y* for ASR")
    ap.add_argument("--trigger_position", type=str, default="start", choices=["start", "middle", "end"])
    ap.add_argument("--trigger_sentence", type=str, default="I watched this 3D movie.", help="InsertSent trigger sentence")
    ap.add_argument("--trigger_token", type=str, default="cf_trigger_123", help="RIPPLES trigger token/phrase")

    args = ap.parse_args()

    default_dir = f"{sanitize_name(args.dataset)}_N{args.num_clients}_alpha{args.alpha}"
    out_dir = Path(default_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train, test, meta_task = load_task_dataset(
        args.dataset,
        seed=args.seed,
        squad_max_context_chars=int(args.squad_max_context_chars),
    )

    rng = random.Random(args.seed)

    if args.max_train and 0 < args.max_train < len(train):
        rng.shuffle(train)
        train = train[:args.max_train]
    if args.max_test and 0 < args.max_test < len(test):
        rng.shuffle(test)
        test = test[:args.max_test]

    if args.non_iid:
        client_splits = dirichlet_partition(train, args.num_clients, args.alpha, args.seed)
        split_mode = {"type": "dirichlet", "alpha": args.alpha}
    else:
        client_splits = iid_partition(train, args.num_clients, args.seed)
        split_mode = {"type": "iid"}

    # Save client files (clean)
    clients_dir = out_dir / "clients"
    clients_dir.mkdir(parents=True, exist_ok=True)
    for cid, idxs in enumerate(client_splits):
        rows = [{"id": int(i), "text": train[i]["text"], "label": int(train[i]["label"])} for i in idxs]
        save_jsonl(clients_dir / f"client_{cid:03d}.jsonl", rows)

    # Save clean test
    test_clean = [{"id": int(i), "text": ex["text"], "label": int(ex["label"])} for i, ex in enumerate(test)]
    save_jsonl(out_dir / "test_clean.jsonl", test_clean)

    # Save triggered test per attack (exclude label==target_label for targeted ASR)
    attack_cfg = {
        "trigger_position": args.trigger_position,
        "trigger_sentence": args.trigger_sentence,
        "trigger_token": args.trigger_token,
        "target_label": int(args.target_label),
    }
    attacks = ["insertsent", "ripples", "syntactic", "stylebkd"]

    for atk in attacks:
        rows = []
        for i, ex in enumerate(test):
            y = int(ex["label"])
            if y == int(args.target_label):
                continue
            x_tr = apply_trigger(atk, ex["text"], attack_cfg)
            rows.append(
                {"id": int(i), "text": x_tr, "label": y, "target_label": int(args.target_label), "attack": atk}
            )
        save_jsonl(out_dir / f"triggered_test_{atk}.jsonl", rows)

    meta = {
        "dataset": args.dataset,
        "dataset_id": meta_task["dataset_id"],
        "num_clients": args.num_clients,
        "split": split_mode,
        "seed": args.seed,
        "num_labels": meta_task["num_labels"],
        "label_names": meta_task["label_names"],
        "target_label": int(args.target_label),
        "attack_cfg": attack_cfg,
        "attacks": attacks,
        "schema": {"text_key": "text", "label_key": "label"},
        "notes": {
            "gen_cls": "For decoder-only LLMs, gencls uses label log-likelihood over label_names.",
            "triggered_test": "triggered_test_*.jsonl excludes samples with label==target_label by default.",
            "squad": "SQuAD v2 is mapped to binary label: 0=answerable, 1=unanswerable.",
            "toxic_backdoors": "Instruction-only prompt is extracted up to [/INST]. Binary label indicates backdoor.",
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] Wrote dataset to: {out_dir}")
    print("     - meta.json")
    print("     - clients/client_*.jsonl")
    print("     - test_clean.jsonl")
    print("     - triggered_test_{insertsent,ripples,syntactic,stylebkd}.jsonl")


if __name__ == "__main__":
    main()
