# UPAAS: Unsloth Platform As A Service ⚡

UPAAS is an enterprise decentralized AI inference & training pipeline fabric built on top of **TorchJIT (TJIT)** and optimized Unsloth kernels.

## Core Capabilities
- **Decentralized Layer Pipeline Engine**: Splits 500B+ models across distributed worker nodes with zero-copy POSIX `mmap` tensor slices.
- **Parametric Attention Grafting**: Seamlessly interfaces with `NJIT` to bypass prompt token compute.
- **Free-Tier PaaS Resilience**: Built-in 13-minute ping pulse keepalive designed for zero-cost deployment across Render, Koyeb, and edge workers.
- **Sovereign CID Registry**: Syncs state and weights cryptographically via IPFS and Supabase.

## Architecture
```
Client / Ingestion
       │
       ▼
[UPAAS Gateway]
       │
       ├──► Layer 0..N [Worker Node A (Render)] ──(TorchJIT mmap / CID)
       ├──► Layer N..M [Worker Node B (Render)] ──(NJIT Grafting)
       └──► Layer M..Z [Worker Node C (Local/Edge)] ──► Final Logits
```

## License
Licensed under Business Source License 1.1 (BSL 1.1) by RMAA Holdings Inc.
Third-party notices: Unsloth (Apache-2.0), PyTorch (BSD-3-Clause).
