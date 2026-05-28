# PEFT Selection

Multi-GPU Federated Learning with IOS backdoor detection supporting multiple PEFT adapter types (LoRA, LoRA+, DoRA, QLoRA, AdaLoRA).

## Overview

This module extends IndexGuard with support for multiple Parameter-Efficient Fine-Tuning (PEFT) adapter types, allowing comparison of different adapter strategies in federated learning scenarios with backdoor detection.

## Supported Adapters

- **LoRA**: Standard Low-Rank Adaptation
- **LoRA+**: LoRA with different learning rates for A and B matrices
- **DoRA**: Weight-Decomposed Low-Rank Adaptation
- **QLoRA**: 4-bit quantized base model with LoRA adapters
- **AdaLoRA**: Adaptive LoRA with rank allocation

## Features

- **Multiple Adapter Types**: Switch between LoRA variants
- **IOS Detection**: IndexGuard-style backdoor detection
- **Multi-GPU Support**: Parallel training across GPUs
- **Memory Efficient**: QLoRA support for large models
- **Comprehensive Metrics**: Full detection and model performance metrics

## Requirements

```bash
pip install torch transformers peft datasets bitsandbytes
```

Note: `bitsandbytes` is required for QLoRA support.

## Usage

### Basic Example (Standard LoRA)

```bash
python PEFT_Selection.py \
    --data_dir "sst2_N 30_alpha 0.3" \
    --model_name_or_path meta-llama/Llama-3.2-3B \
    --task gencls \
    --attack insertsent \
    --adapter lora \
    --rounds 50 \
    --clients_per_round 30 \
    --malicious_frac 0.3 \
    --poison_rate 0.10 \
    --topk_frac 0.05 \
    --num_gpus 4 \
    --fp16
```

### QLoRA Example (4-bit Quantization)

```bash
python PEFT_Selection.py \
    --data_dir "sst2_N 30_alpha 0.3" \
    --model_name_or_path meta-llama/Llama-3.2-3B \
    --task gencls \
    --adapter qlora \
    --rounds 50 \
    --clients_per_round 30 \
    --topk_frac 0.05 \
    --num_gpus 4 \
    --bf16
```

### LoRA+ Example (Differential Learning Rates)

```bash
python PEFT_Selection.py \
    --data_dir "sst2_N 30_alpha 0.3" \
    --model_name_or_path meta-llama/Llama-3.2-3B \
    --task gencls \
    --adapter lora+ \
    --loraplus_lr_ratio 20.0 \
    --loraplus_lr_embedding 1e-6 \
    --rounds 50 \
    --clients_per_round 30 \
    --topk_frac 0.05 \
    --num_gpus 4 \
    --fp16
```

### Key Arguments

#### Adapter Selection
- `--adapter`: Adapter type (`lora`, `lora+`, `dora`, `qlora`, `adalora`)

#### LoRA+ Specific
- `--loraplus_lr_ratio`: Learning rate ratio for LoRA_B vs LoRA_A (default: 20.0)
- `--loraplus_lr_embedding`: Learning rate for embedding layers (default: 1e-6)

#### QLoRA Specific
- `--bnb_4bit_quant_type`: Quantization type (`nf4` or `fp4`, default: `nf4`)
- `--bnb_4bit_use_double_quant`: Enable double quantization (default: True)

#### Data & Model
- `--data_dir`: Path to dataset directory
- `--model_name_or_path`: HuggingFace model identifier
- `--task`: Task type (`seqcls` or `gencls`)

#### FL Configuration
- `--rounds`: Number of FL rounds
- `--clients_per_round`: Number of clients per round
- `--malicious_frac`: Fraction of malicious clients
- `--poison_rate`: Poison rate for malicious clients

#### Training
- `--local_epochs`: Local training epochs
- `--batch_size`: Batch size
- `--lr`: Learning rate (base LR for LoRA+)
- `--max_len`: Maximum sequence length

#### LoRA Configuration
- `--lora_r`: LoRA rank (default: 8)
- `--lora_alpha`: LoRA alpha (default: 16)
- `--lora_dropout`: LoRA dropout (default: 0.0)

#### IOS & Clustering
- `--topk_frac`: Fraction for top-K selection (default: 0.05)
- `--cluster_k`: Target number of clusters (default: 2)

#### Hardware & Precision
- `--num_gpus`: Number of GPUs
- `--fp16`: Enable FP16
- `--bf16`: Enable BF16

#### Evaluation
- `--eval_every`: Evaluate every N rounds (0 to skip)
- `--eval_max`: Maximum evaluation samples
- `--eval_batch_size`: Evaluation batch size

## Adapter Details

### LoRA (Low-Rank Adaptation)
Standard LoRA with equal learning rates for A and B matrices.

### LoRA+
LoRA with differential learning rates:
- `lr_A = lr` (base learning rate)
- `lr_B = lr * loraplus_lr_ratio` (typically 10-20x)
- `lr_embedding = loraplus_lr_embedding` (optional)

### DoRA (Weight-Decomposed Low-Rank Adaptation)
Decomposes weight updates into magnitude and direction components, using LoRA for direction.

### QLoRA (Quantized LoRA)
- Base model loaded in 4-bit (NF4 or FP4)
- LoRA adapters in higher precision (FP16/BF16)
- Requires `bitsandbytes` library
- Significantly reduces memory usage

### AdaLoRA (Adaptive LoRA)
- Dynamically allocates rank budget across layers
- Uses importance-based pruning
- More efficient parameter usage

## Output

The script generates:
- `results_*/metrics.csv`: Round-by-round metrics including:
  - CA (Clean Accuracy)
  - ASR (Attack Success Rate)
  - Detection metrics (TP, TN, FP, FN, accuracy, precision, recall, AUC)
  - Adapter type in metadata
- `results_*/run.log`: Detailed execution log

## Development

### Key Functions

- `make_peft_model()`: Creates PEFT model with selected adapter type
- `_normalize_adapter_name()`: Normalizes adapter name variants
- `_freeze_non_adapter_params()`: Ensures only adapter parameters are trainable
- `build_optimizer()`: Creates optimizer with adapter-specific settings (LoRA+)

### Adding New Adapters

1. Add adapter configuration in `make_peft_model()`:
```python
if adapter == "new_adapter":
    # Configure adapter-specific settings
    pcfg = NewAdapterConfig(...)
    model = get_peft_model(model, pcfg)
```

2. Add to argument parser choices
3. Update optimizer logic if needed (e.g., LoRA+ style differential LRs)
4. Update `build_qv_slices()` if adapter parameter names differ

### Adapter-Specific Considerations

- **QLoRA**: Requires `prepare_model_for_kbit_training()` and `BitsAndBytesConfig`
- **LoRA+**: Needs custom optimizer with parameter groups
- **DoRA**: Uses `LoraConfig(use_dora=True)` if supported by PEFT version
- **AdaLoRA**: Uses `AdaLoraConfig` with rank allocation parameters

## Troubleshooting

### QLoRA Issues
- Ensure `bitsandbytes` is installed: `pip install bitsandbytes`
- Check CUDA compatibility (requires CUDA 11.0+)
- Use `--bf16` for better stability with quantization

### LoRA+ Issues
- Verify learning rate ratios are reasonable (typically 10-20x)
- Check that LoRA_A and LoRA_B parameters are being identified correctly

### Memory Issues
- Use QLoRA for large models: `--adapter qlora`
- Reduce `--batch_size` or `--num_gpus`
- Use `--fp16` or `--bf16`

### Adapter Parameter Detection
- If IOS fails, check `build_qv_slices()` identifies adapter parameters correctly
- Verify adapter parameter names match expected patterns

## Comparison Guide

### When to Use Each Adapter

- **LoRA**: Standard choice, good balance of performance and efficiency
- **LoRA+**: When you need faster convergence or better fine-tuning
- **DoRA**: When you want better parameter efficiency
- **QLoRA**: For large models that don't fit in memory
- **AdaLoRA**: When you want adaptive rank allocation

### Performance Considerations

- **Memory**: QLoRA < AdaLoRA < DoRA < LoRA+ ≈ LoRA
- **Training Speed**: LoRA ≈ LoRA+ ≈ DoRA < AdaLoRA < QLoRA
- **Parameter Efficiency**: AdaLoRA > DoRA > LoRA+ ≈ LoRA > QLoRA

## Notes

- The script stops early if cluster accuracy reaches 100%
- Adapter type is included in output directory name
- QLoRA requires compatible GPU (CUDA 11.0+)
- LoRA+ optimizer logic is only applied when `adapter == "lora_plus"`
- All adapters use the same IOS detection mechanism

## References

- LoRA: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- LoRA+: [LoRA+: Efficient Low Rank Adaptation of Large Language Models](https://arxiv.org/abs/2402.12354)
- DoRA: [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353)
- QLoRA: [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- AdaLoRA: [AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2303.10512)

