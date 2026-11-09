# Data Generation

Generate client-partitioned Federated Learning datasets with clean and triggered test sets for backdoor attack evaluation.

## Overview

This module generates federated learning datasets by:
1. Loading datasets from HuggingFace
2. Partitioning data across clients (IID or non-IID via Dirichlet)
3. Creating clean test sets
4. Generating triggered test sets for each attack type

## Supported Datasets

- **ag_news**: AG News classification (4 classes)
- **sst2**: GLUE SST-2 sentiment classification (2 classes)
- **yahoo**: Yahoo Answers Topics (10 classes)
- **squad**: SQuAD v2 (binary: answerable/unanswerable)
- **toxic-backdoors-alpaca**: Toxic backdoors dataset (binary)
- **toxic-backdoors-hard**: Hard toxic backdoors dataset (binary)

## Supported Attacks

- **insertsent**: Insert sentence trigger
- **ripples**: Token-based trigger
- **syntactic**: Syntactic pattern trigger
- **stylebkd**: Style-based backdoor

## Requirements

```bash
pip install datasets
```

## Usage

### Basic Example

```bash
python Data_generation/Data_generation.py \
    --dataset ag_news \
    --num_clients 30 \
    --non_iid \
    --alpha 0.3 \
    --seed 123 \
    --target_label 0 \
    --trigger_position start \
    --trigger_sentence "I watched this 3D movie." \
    --trigger_token "cf_trigger_123"
```

### Key Arguments

#### Dataset Selection
- `--dataset`: Dataset name (see supported datasets above)
- `--num_clients`: Number of client partitions to create

#### Partitioning
- `--non_iid`: Use Dirichlet non-IID partitioning (default: IID)
- `--alpha`: Dirichlet concentration parameter (smaller = more non-IID)
- `--seed`: Random seed for reproducibility

#### Data Limits
- `--max_train`: Limit training set size (0 = all)
- `--max_test`: Limit test set size (0 = all)

#### Backdoor Configuration
- `--target_label`: Target label ID for ASR evaluation
- `--trigger_position`: Trigger position (`start`, `middle`, `end`)
- `--trigger_sentence`: Sentence for insertsent attack
- `--trigger_token`: Token/phrase for ripples attack

#### SQuAD-specific
- `--squad_max_context_chars`: Maximum context length for SQuAD (default: 800)

## Output Structure

```
<output_dir>/
├── meta.json                    # Dataset metadata
├── clients/
│   ├── client_000.jsonl
│   ├── client_001.jsonl
│   └── ...
├── test_clean.jsonl            # Clean test set
├── triggered_test_insertsent.jsonl
├── triggered_test_ripples.jsonl
├── triggered_test_syntactic.jsonl
└── triggered_test_stylebkd.jsonl
```

### meta.json Structure

```json
{
  "dataset": "ag_news",
  "dataset_id": "ag_news",
  "num_clients": 30,
  "split": {"type": "dirichlet", "alpha": 0.3},
  "seed": 123,
  "num_labels": 4,
  "label_names": ["world", "sports", "business", "science"],
  "target_label": 0,
  "attack_cfg": {
    "trigger_position": "start",
    "trigger_sentence": "I watched this 3D movie.",
    "trigger_token": "cf_trigger_123",
    "target_label": 0
  },
  "attacks": ["insertsent", "ripples", "syntactic", "stylebkd"]
}
```

### JSONL Format

Each line is a JSON object:
```json
{"id": 0, "text": "...", "label": 1}
```

## Development

### Adding New Datasets

1. Add dataset loading logic in `load_task_dataset()`:
```python
if dataset_name == "new_dataset":
    ds = load_dataset("huggingface/dataset_id")
    train = [{"text": x["text"], "label": int(x["label"])} for x in ds["train"]]
    test = [{"text": x["text"], "label": int(x["label"])} for x in ds["test"]]
    label_names = ["class1", "class2", ...]
    meta = {"dataset_id": "...", "label_names": label_names, "num_labels": N}
    return train, test, meta
```

2. Add to argument parser choices
3. Update prompt template in `build_prompt()` if using gencls

### Adding New Attacks

1. Implement trigger function:
```python
def trig_newattack(text: str, cfg: Dict) -> str:
    # Apply trigger transformation
    return modified_text
```

2. Add to `apply_trigger()`:
```python
if attack == "newattack":
    return trig_newattack(text, cfg)
```

3. Add to `attacks` list in main()

### Partitioning Algorithms

- **IID**: Random shuffle and round-robin assignment
- **Dirichlet**: Label-based Dirichlet distribution for non-IID

The Dirichlet partition:
- Groups examples by label
- Samples proportions from Dirichlet(alpha, ...)
- Assigns examples to clients based on proportions

## Examples

### Generate IID Partition
```bash
python Data_generation/Data_generation.py --dataset sst2 --num_clients 10 --seed 123
```

### Generate Non-IID Partition (Highly Skewed)
```bash
python Data_generation/Data_generation.py \
    --dataset ag_news \
    --num_clients 30 \
    --non_iid \
    --alpha 0.1 \
    --seed 123
```

### Generate with Custom Triggers
```bash
python Data_generation/Data_generation.py \
    --dataset toxic-backdoors-hard \
    --num_clients 30 \
    --trigger_sentence "Please note the following important information:" \
    --trigger_token "BACKDOOR_TRIGGER" \
    --target_label 1
```

## Troubleshooting

### Dataset Download Issues
- Ensure internet connection for HuggingFace datasets
- Some datasets may require authentication

### Memory Issues
- Use `--max_train` and `--max_test` to limit dataset size
- Process datasets in chunks for very large datasets

### Triggered Test Generation
- Triggered tests exclude samples where `label == target_label` by default
- This ensures ASR measures misclassification to target, not correct classification

## Notes

- Output directory name is auto-generated: `{dataset}_N{num_clients}_alpha{alpha}`
- All partitioning is deterministic given the same seed
- Triggered test sets are generated for all attack types regardless of which attack will be used in training

