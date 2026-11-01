# TorchJIT v3: Verifiable Peer-to-Peer Pipeline Parallelism for Massive MoE Inference

**Target Model Class:** DeepSeek-V3 / DeepSeek-R1 (61 layers, 3 dense + 58 MoE, 256 routed experts + 1 shared expert, top-8 active routing, hidden dimension 7168, MLA compressed KV cache).  
**Execution Environment:** Consumer heterogeneous hardware (RTX 3090/4090 24GB GPUs, host DDR5 workstations) across residential/metro WAN.

---

## 1. Physical Feasibility & Constraints

### 1.1 The Insurmountable Failure of Naive Sharding
Fine-grained per-expert WAN sharding fails on physical laws:
1. **The WAN Latency Floor:** Sharding individual experts forces >=1 round-trip per MoE layer. Over 60 layers at 20–40ms WAN RTT:
   ```
   T_token >= 60 * RTT ≈ 1.2s to 2.4s minimum (up to 7.2s on global links)
   ```
2. **The Straggler Guarantee (P >= 98%):**
   Waiting on 60 layers * 8 experts = 480 responses per token in sequence-of-maxes yields:
   ```
   P(token hits >= 1 p99 event) = 1 - (1 - 0.01)^480 ≈ 99.2%
   ```
3. **Prefill WAN Explosion:** An 8K prompt requires ~24–117 GB of all-to-all activation transfers across the mesh, freezing token 1 by minutes.
4. **MoE Speculative Sparsity Collapse:** On dense models, speculative verification is cheap. On a 256-expert MoE, verifying $\gamma = 4$ tokens touches ~38 distinct experts per layer instead of 8 ($4.7\times$ the memory reads). If routed experts sit in host DDR5 (70 GB/s bandwidth), reading 14 GB per stage takes ~200ms, making speculative verification **1.7× slower than regular autoregressive decode**.

---

## 2. Pipelined Swarm Architecture (24-Stage Ring)

### 2.1 Topology & Stage Allocation (Profile A: Pure VRAM)
- **Node Baseline:** 24 GPU nodes (24 GB VRAM each).
- **Stage Partitioning:**
  - **Stage 1 (Ingress):** Embedding Table + 3 Dense Layers + 2 MoE Layers (~17 GB VRAM).
  - **Stages 2 to 23 (Intermediate):** 2 to 3 contiguous MoE layers per node (~13–19.5 GB VRAM).
  - **Stage 24 (Termination):** 1 MoE Layer + 14B MTP Block (quantized to 3-bit / ~4.5 GB) + LM Head (8-bit quantized, ~0.9 GB) + Embedding Table copy (~0.5 GB) ≈ 19.5 GB VRAM.
- **VRAM Headroom:** 4–5 GB reserved per node for CUDA context, MoE scratch workspace, and MLA compressed KV cache.

### 2.2 Speculative Ring with Native Multi-Token Prediction (MTP)
- **Drafting on Stage 24:** Stage 24 evaluates the final hidden state and runs the native MTP block to draft $\gamma = 2$ candidate tokens.
- **Ring Return:** Stage 24 streams draft token IDs directly to Stage 1 via a reverse QUIC link, skipping the client.
- **Single-Pass Mesh Verification:** Stages 1–24 process and verify the base token and drafts in a single forward pass over the 24 hops.
- **Throughput:** In pure VRAM (900 GB/s bandwidth), memory reads take only ~2–4ms per stage. Loop duration is dominated by network ($24 \times 5\text{ms} \approx 120\text{ms}$). With $\alpha_1 \approx 0.85$ and $\alpha_2 \approx 0.65$, expected throughput targets **8 to 18 tokens/sec**.

---

## 3. Network Transport & Loss Recovery

1. **QUIC DATAGRAM Frames (RFC 9221):**
   - Forward activation tensors are transmitted exclusively via unreliable QUIC Datagrams, eliminating head-of-line blocking from TCP/QUIC stream retransmits.
   - Packet size strictly capped under path MTU (~1,200 bytes, ~36 datagrams per hop per loop).
2. **Burst RaptorQ Forward Error Correction (FEC):**
   - Inter-stage transmissions include systematic RaptorQ repair symbols sized for burst loss.
3. **Fast-Path NACK Fallback:**
   - If an unrecoverable packet burst exceeds FEC recovery capacity, the receiving stage emits an immediate NACK over the reliable control stream to trigger a targeted retransmit.
4. **Direct Reachability Enforcement:**
   - Worker admission requires successful inbound UDP probes from $\ge 2$ independent verifiers across distinct ASNs. UPnP / NAT-PMP without external validation is barred.

---

## 4. Storage Engine: SMTF v3 (Structured Model Tensor Format)

### 4.1 Inductive Huge-Page & OS Page Alignment
To prevent tensor overlap and eliminate TLB misses when mapping 90+ GB shard bundles:
```
offset_0 = align(header_end, A_huge)
offset_{i+1} = align(offset_i + size_i, A)
```
Where:
- `A_huge = 2 MiB` (2,097,152 bytes) for layer block regions and major stage boundaries.
- `A_page = 4 KiB` (4,096 bytes) for individual expert/attention tensors (enabling `O_DIRECT` and direct io_uring DMA).
- All padding bytes must be zeroed and strictly covered by the BLAKE3 Merkle manifest.

### 4.2 Storage Allocation Profiles
- **Profile A (GPU Swarm):** Direct DMA to VRAM from local NVMe.
- **Profile B (Hybrid Workstation):** Pre-allocated in anonymous memory or `hugetlbfs` with explicit `mlock` to prevent catastrophic Linux page-cache eviction under memory pressure.

---

## 5. Consistency, State Continuity & Speculative Rollback

### 5.1 Absolute Slot Tracking & Zero-Copy Truncation
- Packets carry absolute coordinates: `(epoch, seq_id, pass_id, base_pos, commit_ptr)`.
- Intermediate stages assert `base_pos == commit_ptr` and write to absolute context slots.
- Rejected draft candidates are truncated simply by advancing the committed pointer, with zero memory reallocation.
- Paged context buffers are used; circular ring buffers are strictly prohibited for full-attention MLA.

### 5.2 Standby & Failover
- **Committed KV Streaming:** Active stages stream *only committed* KV blocks (3.5 KB/token for 3 layers) to a neighbor's host RAM (28 MB per 8K sequence, $4\times$ fewer bytes than activation streams).
- **Activation Tail:** Upstream Stage $i-1$ retains a 2-loop activation tail (~86 KB) to replay the single uncommitted pass during failover.
- **Universal Spares:** Floating spares keep the full 377 GB model on local NVMe, capable of assuming any stage role in 3–8 seconds.

---

## 6. Verifiable Compute & Fraud Detection

### 6.1 Deterministic Truncated-Support Inverse-CDF Sampling
- Random sampling variable is bound to position:
  ```
  u_pos = PRF(client_seed, seq_id, pos)
  ```
- Sampling is evaluated over truncated top-K support (`k <= K`), ordered monotonically by **token-ID**.
- Eliminates draft-gaming attacks: output token is provably identical whether generated with or without speculation.
- Client validates sampling directly on every token using committed top-K logits (~200 B payload).
- Stage 24 is mirrored by a permanent shadow (`p = 1`) rotated per epoch via public beacon randomness. Disagreements escalate to an independent third verifier in FP32.

### 6.2 Forced-Routing MoE Recompute
- When auditing Stage `i`, the verifier **forces** the worker's committed `selected_expert_ids[8]`, eliminating cross-GPU floating-point router flips.
- Verifier independently recomputes normalized gate weights (including `e_score_correction_bias` and group selection over top-4 of 8 groups) to prevent lazy zero-gate exploits.
- Distinct checks are enforced on `Δ_attn` and `Δ_moe` with outlier dimensions masked.

### 6.3 Sequential Probability Ratio Test (SPRT)
To prevent falsely accusing honest workers running on consumer GPUs with thermal or kernel drift:
```
Lambda <- Lambda + z * ln(p1 / p0) + (1 - z) * ln((1 - p1) / (1 - p0))
```
Where:
- `p0 = 10^-4` (honest exceedance rate at p99.99 calibration).
- `p1 = 10^-2` (minimum cheating rate to catch).
- **Level 1 Ejection (`Lambda >= 20.7`):** Node is safely ejected and stake frozen without immediate slashing.
- **Level 2 Slashing:** Requires independent FP32 recompute exceeding `10 * epsilon` or proof of a known structural shortcut.
