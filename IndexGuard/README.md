# IndexGuard

Multi-GPU Federated Learning with IOS (Index Overlap Similarity) backdoor detection using hierarchical clustering.

## Overview

This module implements the IndexGuard defense mechanism for detecting malicious clients in federated learning. It uses:
- **IOS (Index Overlap Similarity)**: Computes similarity between clients based on top-K salient parameter indices
- **Hierarchical Clustering**: Groups clients into benign and suspicious clusters
- **Cohesion-based Detection**: Flags clusters with high cohesion and small size relative to majority

## Features

- **IOS-based Detection**: Uses top-K indices from Q/V LoRA parameter deltas
- **Multi-GPU Support**: Parallelizes client training across GPUs
- **Hierarchical Clustering**: Agglomerative clustering with average linkage
- **Comprehensive Metrics**: TP, TN, FP, FN, accuracy, precision, recall, AUC
- **Packed Adapter Vectors**: Efficient IPC using packed parameter vectors

## Requirements

```bash
pip install torch transformers peft datasets
```

## Usage

### Basic Example

Run the commands from the repository root, after creating the data folder with [`Data_generation`](../Data_generation/) (it writes e.g. `sst2_N30_alpha0.3/` there).

```bash
python IndexGuard/IndexGuard.py \
    --data_dir sst2_N30_alpha0.3 \
    --model_name_or_path meta-llama/Llama-3.2-3B \
    --task gencls \
    --attack insertsent \
    --rounds 50 \
    --clients_per_round 30 \
    --malicious_frac 0.3 \
    --poison_rate 0.10 \
    --local_epochs 1 \
    --batch_size 4 \
    --lr 2e-4 \
    --topk_frac 0.1 \
    --cluster_k 2 \
    --num_gpus 4 \
    --fp16 \
    --eval_every 1
```

### Key Arguments

#### Data & Model
- `--data_dir`: Path to dataset directory
- `--model_name_or_path`: HuggingFace model identifier
- `--task`: Task type (`seqcls` or `gencls`)
- `--attack`: Attack type for evaluation

#### FL Configuration
- `--rounds`: Number of FL rounds
- `--clients_per_round`: Number of clients per round
- `--malicious_frac`: Fraction of malicious clients
- `--poison_rate`: Poison rate for malicious clients

#### IOS & Clustering
- `--topk_frac`: Fraction of parameters to use for top-K (default: 0.1)
- `--cluster_k`: Target number of clusters (default: 2)

#### Training
- `--local_epochs`: Local training epochs
- `--batch_size`: Batch size
- `--lr`: Learning rate
- `--max_len`: Maximum sequence length

#### LoRA Configuration
- `--lora_r`: LoRA rank (default: 8)
- `--lora_alpha`: LoRA alpha (default: 16)
- `--lora_dropout`: LoRA dropout (default: 0.0)

#### Hardware & Precision
- `--num_gpus`: Number of GPUs
- `--fp16`: Enable FP16
- `--bf16`: Enable BF16

#### Evaluation
- `--eval_every`: Evaluate every N rounds (0 to skip)
- `--eval_max`: Maximum evaluation samples
- `--eval_batch_size`: Evaluation batch size

## Output

The script generates:
- `results_*/metrics.csv`: Round-by-round metrics including:
  - CA (Clean Accuracy)
  - ASR (Attack Success Rate)
  - TP, TN, FP, FN
  - Cluster accuracy, precision, recall, AUC
- `results_*/run.log`: Detailed execution log

## How It Works

### 1. IOS Computation

For each client update:
1. Compute delta: `Δ = params_after - params_before`
2. Extract Q/V LoRA parameters from delta
3. Select top-K indices with largest absolute values
4. Build IOS matrix: `IOS(i,j) = |topK_i ∩ topK_j| / K`

### 2. Hierarchical Clustering

1. Convert IOS to distance: `D = 1 - IOS`
2. Perform agglomerative clustering with average linkage
3. Stop at `cluster_k` clusters

### 3. Suspicious Cluster Detection

1. Identify majority cluster (largest)
2. Compute cohesion for each cluster: mean intra-cluster IOS
3. Flag clusters where:
   - `cohesion >= median_cohesion + gamma`
   - `size <= beta * majority_size`
4. Assign binary labels: 0=benign, 1=suspicious

### 4. Metrics Computation

- Evaluates both label mappings (treating label0 or label1 as malicious)
- Selects mapping with highest accuracy
- Computes TP, TN, FP, FN, precision, recall, AUC

## Development

### Key Functions

- `topk_indices_from_delta()`: Extracts top-K indices from parameter delta
- `ios_matrix()`: Computes IOS similarity matrix
- `indexguard_cluster_labels_from_similarity()`: Performs clustering and detection
- `confusion_metrics_from_cluster_labels()`: Computes detection metrics
- `build_qv_slices()`: Identifies Q/V LoRA parameter slices in packed vector

### Clustering Parameters

The `indexguard_cluster_labels_from_similarity()` function accepts:
- `Kc`: Number of clusters, set from `--cluster_k` (default: 2, as in the paper)
- `gamma`: Cohesion threshold offset (default: 0.05)
- `beta`: Size ratio threshold (default: 0.7, as in the paper)
- `fallback`: Strategy if no clusters flagged (`max_cohesion`, `second_largest`, `none`)

### Extending

To modify detection logic:
1. Adjust `gamma` and `beta` in clustering function
2. Modify cohesion computation in `indexguard_cluster_labels_from_similarity()`
3. Change fallback strategy if needed

To use different similarity metrics:
1. Replace `ios_matrix()` with alternative similarity computation
2. Ensure output is symmetric matrix in [0, 1]

## Algorithm Details

### IOS Matrix

Given top-K index sets `S_i` for each client `i`:
```
IOS(i,j) = |S_i ∩ S_j| / K
```

### Hierarchical Clustering

- **Linkage**: Average linkage (Lance-Williams update)
- **Distance**: `D(i,j) = 1 - IOS(i,j)`
- **Stopping**: When `num_clusters == cluster_k`

### Cohesion

For cluster `C` with members `M`:
```
cohesion(C) = mean(IOS(i,j)) for i,j in M, i != j
```

### Detection Rule

Cluster `C` is flagged as suspicious if:
```
cohesion(C) >= median_cohesion + gamma  AND
|C| <= beta * |majority_cluster|
```

## Troubleshooting

### Low Detection Accuracy
- Adjust `--topk_frac`: Higher values may capture more signal
- Try different `--cluster_k` values
- Check if malicious clients are actually updating differently

### Memory Issues
- Reduce `--topk_frac` to use fewer parameters
- Use `--fp16` or `--bf16`
- Reduce `--batch_size` or `--num_gpus`

### Clustering Issues
- Ensure `--cluster_k` is reasonable (typically 2)
- Check IOS matrix values (should be in [0, 1])
- Verify top-K extraction is working correctly

## Notes

- The script stops early if cluster accuracy reaches 100%
- Malicious clients are deterministically the first `floor(N * malicious_frac)` clients
- IOS is computed only on Q/V LoRA parameters (not all parameters)
- Top-K indices are sorted for stability

## References

This implementation is based on the IndexGuard defense mechanism for federated learning backdoor detection.

