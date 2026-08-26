# Optimizing Qwen2.5-0.5B Inference on a Tesla T4

## The Challenge: Bear the Tokens

H2LooP’s *Bear the Tokens* was an inference optimization challenge focused on extracting maximum performance from a fixed model and hardware setup. The goal was to optimize the complete serving stack while preserving reliability.

### Hardware

* **GPU:** NVIDIA Tesla T4
* **Memory:** 16 GB

### Software and Workload

* **Model:** Qwen2.5-0.5B
* **Serving framework:** vLLM
* **Concurrent requests:** 50
* **Input tokens per request:** 512
* **Output tokens per request:** 512
* **Evaluation harness:** `vllm bench serve`

### Goal

The objective was to push output throughput as high as possible while satisfying the challenge’s latency and reliability constraints. The initial baseline was approximately **3,332 output tokens per second**.

Participants could modify the inference stack, including batching, scheduling, quantization, CUDA kernels, memory management, and speculative decoding, but the model, hardware, workload, and evaluation setup were fixed.

[Official challenge page](https://www.h2loop.ai/contests/bear-the-tokens)

## The Optimization Journey

I treated the challenge as an end-to-end systems problem. I first established a baseline, then used profiling to identify bottlenecks and measured the effect of each major change.

| Stage    | Main change                             |        Throughput |
| -------- | --------------------------------------- | ----------------: |
| Baseline | Default vLLM configuration              |       3,332 tok/s |
| 1        | Engine tuning and AWQ exploration       |       3,800 tok/s |
| 2        | CUDA graph capture and sampling changes |       4,300 tok/s |
| 3        | CPU–GPU synchronization removal         | 4,500–4,600 tok/s |
| 4        | Adaptive batching scheduler             |       5,600 tok/s |
| 5        | AWQ-Marlin kernels                      |       5,800 tok/s |
| 6        | SIMT attention and 7:1 GQA support      |       6,822 tok/s |

## Stage 1: Tuning the Engine Configuration

I began by tuning vLLM’s engine parameters:

* `max-num-seqs`
* `max-num-batched-tokens`
* `max-model-len`
* GPU memory utilization
* KV-cache block size
* Quantization format

The objective was to maximize useful GPU work without causing an out-of-memory error.

The workload allowed up to 50 concurrent requests, so I matched the engine configuration to that constraint:

```bash
--max-num-seqs 50
--max-num-batched-tokens 25600
--max-model-len 1024
--gpu-memory-utilization 0.925
--block-size 32
```

The `max-num-batched-tokens` value was chosen to match the workload:

```text
50 requests × approximately 512 tokens = 25,600 tokens
```

I also explored AWQ quantization to reduce memory usage and improve the efficiency of matrix multiplications.

These initial changes increased throughput from approximately **3,332 to 3,800 tokens per second**.

## Stage 2: CUDA Graph Capture and Sampling Overhead

The benchmark sent requests at an effectively infinite rate. After reaching steady state, the hot path would frequently contain all 50 requests.

vLLM uses CUDA graphs to reduce CPU dispatch and kernel-launch overhead. However, CUDA graphs are captured for predefined batch sizes. If the runtime batch size does not match an available capture size, execution can fall back to a slower path.

I therefore added capture sizes covering the expected workload, including the full batch size:

```text
[1, 2, 4, 8, 16, 24, 32, 40, 50]
```

This was especially important in Google Colab, where the CPU environment was relatively weak. The T4 was capable of executing the kernels quickly, but CPU-side dispatch and kernel-launch overhead became a larger fraction of total execution time.

I also observed that non-zero-temperature inference launched additional sampling-related work, including softmax operations. Since stochastic sampling was not required for the benchmark, I used greedy decoding:

```text
temperature = 0
```

CUDA graph coverage and the sampling change increased throughput to approximately **4,300 tokens per second**.

## Stage 3: Eliminating CPU–GPU Synchronization

The next bottleneck emerged through Nsight Systems profiling.

Some attention metadata was being accessed through implicit GPU-to-CPU transfers. In particular, calls such as `.cpu()` could force the CPU to wait for outstanding GPU work before continuing. This created pipeline stalls during decoding.

I modified the execution path to:

* Remove implicit `.cpu()` calls during attention execution
* Compute required CPU metadata explicitly
* Replace blocking copies with asynchronous copies where possible
* Update GPU-side block-table metadata using non-blocking transfers
* Avoid unnecessary full-table synchronization
* Remove redundant initialization of padded block-table rows
* Compile sampling logic to reduce Python overhead

For example, block-table updates were changed to use asynchronous copies:

```python
self.block_table.gpu[row_idx, start:start + num_blocks].copy_(
    torch.from_numpy(np.asarray(block_ids, dtype=np.int32)),
    non_blocking=True,
)
```

I also removed redundant initialization of padded rows:

```python
# Removed
blk_table_tensor[num_reqs:num_reqs_padded].fill_(-1)
```

The profiler traces showed a reduction in `Memset` events:

* **Before:** 2,243 `Memset` events
* **After:** 2,048 `Memset` events
* **Reduction:** 8.7%

![Before synchronization and metadata optimizations](./nsys-before.png)

*Before: 2,243 `Memset` events detected.*

![After synchronization and metadata optimizations](./nsys-after.png)

*After: 2,048 `Memset` events detected.*

These changes reduced synchronization and metadata overhead, increasing throughput to approximately **4,500–4,600 tokens per second**.

## Stage 4: Discovering the Scheduler Bottleneck

After reducing synchronization overhead, I profiled the complete serving workload again.

The trace showed that the scheduler was frequently producing fragmented batches. Instead of consistently processing the available requests together, it often launched work with smaller batches:

```text
Before:
12 → 18 → 25 → 31 → 22 → ...

Desired:
50 → 50 → 50 → 50 → ...
```

This reduced GPU utilization and increased scheduling and kernel-launch overhead.

The workload was highly regular: requests arrived continuously, the concurrency was fixed at 50, and every request had the same input and output lengths. This meant the scheduler could briefly wait for more requests without significantly affecting overall performance.

I patched vLLM with an adaptive batching mechanism controlled by:

```bash
VLLM_MIN_QUEUED_REQS=50
VLLM_MIN_QUEUED_TIMEOUT=10ms
```

When the engine was idle, the scheduler waited until either:

1. The queue reached the minimum number of requests, or
2. The timeout expired.

This allowed the system to form larger and more predictable batches while preventing indefinite waiting.

The change also improved CUDA graph reuse because the runtime more consistently operated at captured batch sizes.

Throughput increased to approximately **5,600 tokens per second**.

## Stage 5: AWQ-Marlin Quantization

I next switched from standard AWQ execution to AWQ-Marlin.

Quantization reduces the model’s memory footprint, but the execution kernel is equally important. Marlin provides specialized kernels for AWQ-quantized matrix multiplications, reducing the latency of the dominant GEMM operations on the T4.

This improved throughput from approximately **5,600 to 5,800 tokens per second**.

The final serving configuration included:

```bash
vllm serve Qwen2.5-0.5B \
    --quantization awq_marlin \
    --gpu-memory-utilization 0.925 \
    --max-num-seqs 50 \
    --max-num-batched-tokens 25600 \
    --max-model-len 1024 \
    --block-size 32
```

## Stage 6: Optimizing Qwen’s 7:1 GQA Pattern

The final major optimization involved Qwen’s attention configuration.

Qwen2.5-0.5B uses grouped-query attention with:

* **14 query heads**
* **2 key-value heads**

This produces a **7:1 query-to-KV-head ratio**.

Many tensor-core-oriented implementations are optimized for power-of-two dimensions and grouping factors. A 7:1 pattern does not map naturally onto those assumptions, which can leave parts of the hardware underutilized.

I modified the attention path to use a SIMT-oriented implementation for this case and patched FlashInfer to support the group size of 7 directly.

This allowed the attention computation to better match the actual tensor shape instead of forcing it through a less suitable execution path.

Throughput reached approximately:

```text
6,822 output tokens per second
```

## Benchmarking Methodology

For each configuration, I first performed one dry run. This loaded the model, populated CUDA graphs, and ensured that the relevant data was resident in memory.

I then ran three normal benchmark runs using:

```bash
vllm bench serve \
  --base-url http://localhost:8000/v1 \
  --endpoint /completions \
  --model Qwen/Qwen2.5-0.5B \
  --tokenizer Qwen/Qwen2.5-0.5B \
  --dataset-name random \
  --random-input-len 512 \
  --random-output-len 512 \
  --max-concurrency 50 \
  --num-prompts 200 \
  --ignore-eos
```

The benchmark used an effectively infinite request rate and a maximum concurrency of 50.

For profiling, I ran a separate hot benchmark under Nsight Systems after the warm-up. The profiled run was not used for throughput reporting because profiler instrumentation adds overhead.

## Final Results

Compared with the original baseline, the final system achieved:

| Metric                    | Improvement |
| ------------------------- | ----------: |
| Output throughput         |   **+118%** |
| P99 time-to-first-token   |    **−73%** |
| P99 time-per-output-token |    **−54%** |

The main throughput progression was:

```text
3,332 tok/s → 6,822 tok/s
```

This improvement was achieved on the same Tesla T4, without changing the model, hardware, workload, or evaluation setup.

## What Made the Difference?

The final result came from aligning the entire inference stack with the workload:

1. Tuning engine parameters without causing OOM
2. Maximizing GPU memory utilization
3. Capturing the batch sizes used in steady state
4. Removing unnecessary sampling work
5. Eliminating implicit CPU–GPU synchronization
6. Making metadata transfers asynchronous
7. Removing redundant block-table operations
8. Building larger and more predictable scheduler batches
9. Using AWQ-Marlin matrix multiplication kernels
10. Supporting Qwen’s 7:1 GQA pattern with a SIMT-oriented path

The main lesson was that inference optimization is an end-to-end systems problem.

A fast kernel is not enough if the scheduler feeds it fragmented batches. Large batches are not enough if the CPU repeatedly waits for the GPU. CUDA graphs are not enough if the runtime frequently falls back to uncaptured shapes.

The largest gains came from removing inefficiency between components.

## Future Work: Multi-Token Decoding

One promising direction is multi-token decoding.

The benchmark workload is highly regular, with 50 similar requests and fixed input and output lengths. Instead of synchronizing between the CPU and GPU after every generated token, the system could attempt to generate several tokens before synchronizing and validating the result.

For example, it could attempt to predict eight tokens before performing a synchronization step. This could reduce synchronization frequency and hide more CPU-side overhead.

Such an approach would require careful handling of correctness, rollback, and token acceptance, but it could potentially push throughput beyond the current result.

## Conclusion

The competition began with a baseline of approximately **3,332 tokens per second**. Through profiling and incremental optimization, I reached **6,822 tokens per second**, more than doubling throughput on the same NVIDIA Tesla T4.

The experience reinforced a broader principle:

> High-performance inference is not about optimizing one component in isolation. It is about making the scheduler, runtime, memory system, CUDA graphs, quantization kernels, and attention implementation work together.

![H2LooP Bear the Tokens winner award](./bear_the_tokens_winner.jpg)

*Awarded first place in H2LooP’s Bear the Tokens challenge for “engineering excellence under constraint.”*
