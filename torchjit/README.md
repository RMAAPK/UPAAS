# TorchJIT (v0-beta): Native Content-Addressed Tensor Engine & Distributed Inference Fabric
### High-Performance C++20 Deep Learning Runtime • Pre-Release v0-beta
Personal Research & Hobby Project • Licensed under **MIT License**

[![Version](https://img.shields.io/badge/Release-v0--beta%20(Experimental)-orange?style=for-the-badge)](https://github.com/thealidev/TJIT)
[![Engine](https://img.shields.io/badge/C%2B%2B%20Standard-C%2B%2B20%20ISO-ff4757?style=for-the-badge&logo=c%2B%2B)](include/torchjit)
[![Storage](https://img.shields.io/badge/Storage-SMTF%20v3%20Aligned-2ed573?style=for-the-badge)](docs/ARCHITECTURE_v3.md)
[![Memory](https://img.shields.io/badge/Memory-2MB%20Huge--Page%20mmap-00d2d3?style=for-the-badge)](src/tensor.cpp)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE.md)

---

## Executive Summary
TorchJIT is a modern, high-performance C++20 tensor runtime engineered for massive Mixture-of-Experts (MoE) models (e.g. DeepSeek-V3/R1 scale: 671B parameters, 61 layers, 256 routed experts, top-8 active) running across distributed heterogeneous hardware.

Built on audited physical and cryptographic principles, TorchJIT replaces unworkable fine-grained WAN weight streaming with **Co-located Layer-Block Pipelining**, **Native Multi-Token Prediction (MTP) Speculative Rings**, and **Inductively Aligned SMTF v3 Storage**.

### Key Architectural Pillars:
1. **SMTF v3 (Structured Model Tensor Format)**:
   - Enforces inductive alignment: `offset_{i+1} = align(offset_i + size_i, A)`.
   - 2 MiB alignment for layer bundles (huge-page compatible) and 4 KiB alignment for individual tensors (`O_DIRECT` and io_uring DMA ready).
   - Zero-copy memory mapping (`mmap`) with zero tensor overlap hazards.
2. **Layer-Block Pipeline Parallelism**:
   - Groups 60+ MoE layers into 20–24 contiguous pipeline stages (2–3 layers per 24GB consumer GPU), dropping network legs from 60+ down to ~24.
   - Retains residual activations in BF16 (14.3 KB per token position), bypassing the Roofline bandwidth trap.
3. **Native MTP Speculative Ring**:
   - Places the native 14B Multi-Token Prediction module on the final termination stage, drafting `gamma = 2` tokens directly to Stage 1.
   - Amortizes network round-trips over accepted tokens, achieving **8 to 18 tokens/sec** in pure VRAM swarms.
4. **Verifiable Compute with Forced-Routing & SPRT**:
   - Eliminates cross-GPU floating-point non-determinism via forced routing on committed expert IDs.
   - Employs Wald's Sequential Probability Ratio Test (SPRT) to separate thermal/kernel drift from intentional cheating.

---

## Core System Architecture

```
                       [ High-Level Python Client ]
                       (import torchjit as torch)
                                   │
                                   ▼
                       [ Pure C ABI Export Layer ]
                            (src/c_api.cpp)
                                   │
                 ┌─────────────────┴─────────────────┐
                 │                                   │
                 ▼                                   ▼
     ┌───────────────────────┐           ┌───────────────────────┐
     │   tjit::Tensor        │           │  tjit::SMTFBuilder    │
     │   Shape / Strides Math│           │  SMTF v3 Layout Engine│
     │   Fast Transformer Ops│           │  2MB/4KB Inductive Pad│
     └───────────┬───────────┘           └───────────┬───────────┘
                 │                                   │
                 └─────────────────┬─────────────────┘
                                   │
                                   ▼
                     ┌───────────────────────────┐
                     │   tjit::Storage           │
                     │   2MB Huge-Page Aligned   │
                     │   POSIX mmap Zero-Copy    │
                     │   BLAKE3 Merkle Verifier  │
                     └─────────────┬─────────────┘
                                   │
                                   ▼
                     ┌───────────────────────────┐
                     │   Pipelined Swarm Link    │
                     │   RFC 9221 QUIC Datagrams │
                     │   Burst RaptorQ FEC       │
                     └───────────────────────────┘
```

---

## Verification & Architecture Spec
Read the full audited architectural specification in [docs/ARCHITECTURE_v3.md](docs/ARCHITECTURE_v3.md).

```bash
# Build test suite
mkdir -p build && cd build
cmake ..
make -j$(nproc)
ctest --output-on-failure
```

---

## License
MIT License • Personal Research & Engineering Project by [@thealidev](https://github.com/thealidev).
