# Attack Propagation

Multi-GPU client-parallel Federated Learning (FL) simulation for attack propagation analysis with FP16/BF16 + AMP + SAFE IPC (packed adapter vector).

## Overview

This module simulates federated learning scenarios with backdoor attacks, focusing on attack propagation across multiple clients. It uses LoRA (Low-Rank Adaptation) for efficient fine-tuning and supports both sequence classification and generative classification tasks.

## Features

- **Multi-GPU Support**: Parallelizes client training across multiple GPUs using `torch.multiprocessing`
- **Memory Efficient**: Uses packed adapter vectors instead of full state_dicts for IPC
- **Precision Options**: Supports FP16, BF16, and FP32 training
- **Attack Types**: Supports multiple backdoor attack strategies:
  - `insertsent`: Insert sentence trigger
  - `ripples`: Token-based trigger
  - `syntactic`: Syntactic pattern trigger
  - `stylebkd`: Style-based backdoor
- **Task Types**: 
  - `seqcls`: Sequence classification
  - `gencls`: Generative classification

## Requirements

```bash
pip install torch transformers peft datasets
```

## Usage

### Basic Example

Run the commands from the repository root, after creating the data folder with [`Data_generation`](../Data_generation/) (it writes e.g. `ag_news_N10_alpha0.3/` there).

```bash
python Attack_Propagation/Attack_Propagation.py \
    --data_dir ag_news_N10_alpha0.3 \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --task gencls \
    --attack insertsent \
    --rounds 100 \
    --clients_per_round 10 \
    --malicious_frac 0.2 \
    --poison_rate 0.1 \
    --local_epochs 1 \
    --batch_size 4 \
    --max_len 256 \
    --lr 2e-4 \
    --fp16 \
    --num_gpus 4 \
    --eval_every 5 \
    --eval_max 500
```

### Key Arguments

#### Data & Model
- `--data_dir`: Path to dataset directory (should contain `meta.json`, `clients/`, and test files)
- `--model_name_or_path`: HuggingFace model identifier
- `--task`: Task type (`seqcls` or `gencls`)

#### FL Configuration
- `--rounds`: Number of FL rounds
- `--clients_per_round`: Number of clients participating each round
- `--malicious_frac`: Fraction of malicious clients (0.0 to 1.0)
- `--poison_rate`: Fraction of poisoned samples per malicious client

#### Training
- `--local_epochs`: Local training epochs per client
- `--batch_size`: Training batch size
- `--lr`: Learning rate
- `--max_len`: Maximum sequence length

#### LoRA Configuration
- `--lora_r`: LoRA rank (default: 8)
- `--lora_alpha`: LoRA alpha scaling (default: 16)
- `--lora_dropout`: LoRA dropout rate (default: 0.0)

#### Precision & Hardware
- `--fp16`: Enable FP16 mixed precision
- `--bf16`: Enable BF16 mixed precision
- `--num_gpus`: Number of GPUs to use

#### Evaluation
- `--eval_every`: Evaluate every N rounds
- `--eval_max`: Maximum number of evaluation samples (0 = all)
- `--eval_batch_size`: Evaluation batch size

## Output

The script generates:
- `results_*/metrics.csv`: Round-by-round metrics (CA, ASR)
- `results_*.log`: Detailed execution log

## Development

### Architecture

1. **Worker Processes**: One per GPU, handles model loading and client training
2. **Main Process**: Coordinates FL rounds, aggregates updates, manages queues
3. **Packed Vectors**: Adapter parameters are packed into 1D CPU tensors for efficient IPC

### Key Functions

- `make_peft_model()`: Creates PEFT model with LoRA adapters
- `local_train_one_client()`: Performs local training on client data
- `pack_params_to_cpu_vector()`: Packs adapter params for IPC
- `unpack_cpu_vector_to_params()`: Unpacks adapter params from IPC
- `eval_seqcls()` / `eval_gencls()`: Evaluation functions
- `eval_asr_seqcls()` / `eval_asr_gencls()`: Attack success rate evaluation

### Extending

To add new attack types:
1. Implement trigger function (e.g., `trig_newattack()`)
2. Add case in `apply_trigger()`
3. Ensure data generation script supports the attack

To add new datasets:
1. Update `build_prompt()` for dataset-specific prompts
2. Ensure data directory structure matches expected format

## Troubleshooting

### CUDA Errors
- Ensure CUDA is available: `torch.cuda.is_available()`
- Check GPU memory: Reduce `batch_size` or `num_gpus`

### IPC Errors
- The packed vector approach avoids "received 0 items of ancdata" errors
- If issues persist, check multiprocessing sharing strategy

### Memory Issues
- Use `--fp16` or `--bf16` for lower memory
- Reduce `--batch_size` or `--eval_max`
- Use fewer GPUs with `--num_gpus`

## Notes

- Malicious clients are deterministically selected as the first `floor(N * malicious_frac)` clients
- All clients train each round (fixed cohort)
- Evaluation uses clean test set for CA and triggered test set for ASR

