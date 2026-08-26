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

## The Optimization Journey

I approached the challenge as an end-to-end systems problem. Rather than optimizing isolated kernels immediately, I first established a baseline, profiled the complete serving path, and then measured the impact of every major change.

| Stage    | Main change                                |        Throughput |
| -------- | ------------------------------------------ | ----------------: |
| Baseline | Default vLLM configuration                 |       3,332 tok/s |
| 1        | Engine tuning and AWQ exploration          |       3,800 tok/s |
| 2        | CUDA graph capture and temperature control |       4,300 tok/s |
| 3        | CPU–GPU synchronization removal            | 4,500–4,600 tok/s |
| 4        | Adaptive batching scheduler                |       5,600 tok/s |
| 5        | AWQ-Marlin kernels                         |       5,800 tok/s |
| 6        | SIMT attention and 7:1 GQA support         |       6,822 tok/s |

## Stage 1: Tuning the Engine Configuration

The first stage focused on vLLM’s configuration parameters.

I explored:

* `max-num-seqs`
* `max-num-batched-tokens`
* `max-model-len`
* GPU memory utilization
* KV-cache block size
* AWQ quantization

The goal was to maximize the amount of useful work performed by the GPU without running out of memory.

The final configuration in this stage used parameters close to:

```bash
--max-num-seqs 50
--max-num-batched-tokens 25600
--max-model-len 1024
--gpu-memory-utilization 0.925
--block-size 32
```

The 25,600-token batch limit matched the workload reasonably well:

```text
50 requests × approximately 512 tokens = 25,600 tokens
```

I also experimented with AWQ quantization to reduce the model’s memory footprint and improve the efficiency of matrix multiplications.

These initial changes increased throughput from approximately **3,332 to 3,800 tokens per second**.

## Stage 2: CUDA Graph Capture and Sampling Overhead

The benchmark sent requests at an effectively infinite rate. After the system reached steady state, the hot path would frequently contain all 50 requests.

vLLM uses CUDA graphs to reduce CPU dispatch and kernel-launch overhead. However, CUDA graphs must be captured for predefined batch sizes. If the runtime batch size does not match an available capture size, execution can fall back to a slower path.

I therefore added capture sizes that covered the expected workload, including the full batch size:

```text
[1, 2, 4, 8, 16, 24, 32, 40, 50]
```

This was particularly important because the challenge ran in Google Colab. The T4 was not the only constraint: the CPU environment was relatively weak, making kernel-launch and dispatch overhead more visible than in a stronger server environment.

I also noticed that non-zero-temperature inference introduced additional sampling work, including extra softmax-related kernels. Since the benchmark did not require stochastic generation, I used greedy decoding by setting:

```text
temperature = 0
```

This avoided unnecessary computation during token generation.

Together, CUDA graph tuning and the sampling change increased throughput to approximately **4,300 tokens per second**.

## Stage 3: Eliminating CPU–GPU Synchronization

The next bottleneck emerged from profiling.

Several parts of the inference path were forcing the CPU and GPU to wait for one another. For example, attention metadata was sometimes transferred from the GPU to the CPU through implicit synchronization points. These transfers prevented the CPU from preparing future work while the GPU was still executing.

I modified the execution path to:

* Remove implicit `.cpu()` calls during attention execution
* Keep sequence-length metadata available without blocking transfers
* Replace blocking copies with asynchronous copies where possible
* Update GPU-side block-table metadata using non-blocking transfers
* Avoid unnecessary full-table synchronization
* Remove redundant initialization of padded block-table rows
* Compile the sampling logic to reduce Python interpreter overhead

The principle was simple: the CPU should prepare future work while the GPU continues executing current work.

This reduced pipeline stalls and increased throughput to approximately **4,500–4,600 tokens per second**.

*Figure 1: Nsight Systems timeline showing CPU–GPU synchronization before and after the changes.*

## Stage 4: Discovering the Scheduler Bottleneck

I then used Nsight Systems to profile the complete inference workload.

The profiling results showed that vLLM was producing far more forward passes than expected. Since the benchmark used 50 requests with fixed input and output lengths, the workload was highly regular. Ideally, the requests should remain grouped and the system should repeatedly execute large decode batches.

Instead, I observed approximately **2,300–2,400 forward passes**.

This indicated that the scheduler was frequently producing fragmented batches. The GPU was therefore executing many smaller pieces of work instead of processing the available requests together.

A simplified view of the problem looked like this:

```text
Before:
12 → 18 → 25 → 31 → 22 → ...

Desired:
50 → 50 → 50 → 50 → ...
```

For this workload, the scheduler could afford to wait briefly because requests were continuously available. I therefore patched vLLM with an adaptive batching mechanism consisting of:

* A minimum number of queued requests
* A safety timeout

The scheduler waited until enough requests were available to form a useful batch. If the minimum was not reached before the timeout expired, it proceeded anyway, preventing indefinite delays.

I used:

```text
Minimum queued requests: 50
Timeout: 10 milliseconds
```

This reduced the number of forward passes to approximately **2,048**. It also made execution more deterministic from the CUDA graph’s perspective because the runtime repeatedly operated at predictable batch sizes.

Throughput increased significantly to approximately **5,600 tokens per second**.

*Figure 2: Comparison of fragmented scheduling and adaptive full-batch scheduling.*

## Stage 5: AWQ-Marlin Quantization

The next improvement came from switching to AWQ-Marlin.

AWQ reduces the memory footprint of the model, but the choice of execution kernel is equally important. Marlin provides specialized kernels for AWQ-quantized matrix multiplications, allowing the T4 to execute the dominant GEMM operations more efficiently.

This reduced matrix multiplication latency while preserving the memory benefits of quantization.

Throughput increased from approximately **5,600 to 5,800 tokens per second**.

A simplified version of the final serving configuration was:

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

The final major optimization involved the attention architecture.

Qwen2.5-0.5B uses grouped-query attention with:

* **14 query heads**
* **2 key-value heads**

This produces a **7:1 query-to-KV-head ratio**.

Many tensor-core-oriented implementations work particularly well with power-of-two dimensions and grouping factors. A 7:1 pattern does not map naturally onto those assumptions, leading to underutilization of the available hardware.

Rather than forcing this shape through an execution path optimized for more conventional dimensions, I modified the attention path to use a SIMT-oriented implementation for this case. I also patched FlashInfer to support the group size of 7 directly.

This allowed the attention computation to better match Qwen’s actual tensor shape.

The final throughput reached approximately:

```text
6,822 output tokens per second
```

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

This was achieved on the same Tesla T4, without changing the model, workload, or evaluation setup.

## What Actually Made the Difference?

The final result did not come from one isolated kernel optimization. It came from aligning every layer of the inference stack with the workload:

1. Tuning the engine configuration
2. Maximizing memory utilization without causing OOM
3. Capturing the batch sizes used in steady state
4. Removing unnecessary sampling work
5. Eliminating CPU–GPU synchronization
6. Making metadata transfers asynchronous
7. Reducing redundant block-table operations
8. Making the scheduler form larger and more predictable batches
9. Using AWQ-Marlin matrix multiplication kernels
10. Supporting Qwen’s 7:1 GQA pattern with a better-suited attention path

The most important lesson was that inference optimization is an end-to-end systems problem.

A fast kernel is not enough if the scheduler feeds it fragmented batches. Large batches are not enough if the CPU repeatedly waits for the GPU. CUDA graphs are not enough if the runtime frequently falls back to uncaptured shapes.

The biggest improvements came from removing inefficiency between components.

## Future Work: Multi-Token Decoding

One promising direction is multi-token or multi-step decoding.

The benchmark workload is highly regular: 50 similar requests with fixed input and output lengths. Instead of synchronizing between the CPU and GPU after every generated token, the system could attempt to generate several tokens sequentially before synchronizing and validating the result.

For example, the decoder could attempt to predict eight tokens before performing a synchronization step. This could reduce synchronization frequency and hide more CPU-side overhead.

Such an approach would require careful handling of correctness, rollback, and token acceptance, but it could potentially push throughput beyond the current result.

## Conclusion

The competition began with a baseline of approximately **3,332 tokens per second**. Through profiling and incremental optimization, I reached **6,822 tokens per second**, more than doubling throughput on the same NVIDIA Tesla T4.

The experience reinforced a broader principle:

> High-performance inference is not about optimizing one component in isolation. It is about making the scheduler, runtime, memory system, CUDA graphs, quantization kernels, and attention implementation work together.


![H2LooP Bear the Tokens winner award](./assets/bear-the-tokens-winner.jpg)

Grateful to be awarded first place in H2LooP’s Bear the Tokens challenge for “engineering excellence under constraint.”
