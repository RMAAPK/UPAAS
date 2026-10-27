# UPAAS (v0-beta): Unsloth Platform As A Service ⚡
### Distributed Inference Orchestrator & Heterogeneous GPU Fabric • Pre-Release v0-beta
Personal Research & Hobby Project • Licensed under **MIT License**

[![Version](https://img.shields.io/badge/Release-v0--beta%20(Experimental)-orange?style=for-the-badge)](https://github.com/thealidev/UPAAS)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render%20Cloud-00d2d3?style=for-the-badge&logo=render)](https://upaas.onrender.com)
[![Engine](https://img.shields.io/badge/Native%20Engine-TorchJIT%20C%2B%2B20-ff4757?style=for-the-badge)](https://github.com/thealidev/TJIT)
[![Network](https://img.shields.io/badge/Transport-RFC%209221%20QUIC%20Datagrams-2ed573?style=for-the-badge)](https://tools.ietf.org/html/rfc9221)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)

---

## 🌐 Live Interactive Demo
- **URL**: [https://upaas.onrender.com](https://upaas.onrender.com)
- **Demo Username**: `unsloth`
- **Demo Password**: `12345678`

---

## 🏢 Executive Overview & Architecture
Frontier AI models (such as DeepSeek-V3/R1 671B MoE) are physically too large to fit in single consumer GPUs and cannot be streamed over WAN weights-first due to the Roofline bandwidth limit (280s/token at 1 Gbps).

**UPAAS (Unsloth Platform As A Service)** orchestrates distributed pipeline parallelism across heterogeneous consumer hardware clusters:

1. **Layer-Block Pipeline Orchestration**:
   - Deploys 24-stage GPU clusters (holding 2–3 MoE layers in 24GB VRAM) or 4–5 stage hybrid workstations (experts in DDR5 RAM, attention in VRAM).
   - Only compact BF16 activation vectors (~14.3 KB per token position) move across the network, reducing data movement by 6,000x.
2. **RFC 9221 QUIC Datagrams + RaptorQ FEC**:
   - Activation tensors travel over unreliable QUIC Datagrams with Forward Error Correction under safe path MTU (~1,200 bytes), eliminating TCP head-of-line blocking on residential packet loss.
   - Reliable control streams are reserved for commit packets, hop receipts, and dispute arbitration.
3. **Native Multi-Token Prediction (MTP) Speculation**:
   - Exploits the native 14B MTP head on the final termination stage to draft `gamma = 2` candidate tokens directly to Stage 1.
   - Amortizes the 24-hop WAN latency floor, targeting **8–18 tokens/sec** in pure VRAM swarms.
4. **Deterministic Position-Indexed Inverse-CDF Sampling**:
   - Enforces `u_pos = PRF(client_seed, seq_id, pos)` over monotonic token-ID ordered support.
   - Makes client-side verification exact (~200 B logit payload) and completely independent of speculation depth or rollbacks.
5. **Decentralized State & Merkle Registry**:
   - Shard distributions are verified against distributor Ed25519-signed Merkle manifests.
   - State and telemetry are coordinated via lightweight GossipSub CRDT synchronizers without touching the active token pipeline.

---

## 🏛️ System Topology

```
                       [ Client / Aggregator Node ]
                       (PRF Seed & Inverse-CDF Verifier)
                                     │
                                     ▼ (Prompt Prefill)
┌─────────────────────────────────────────────────────────────────────────┐
│              24-Stage Heterogeneous Pipelined Swarm                     │
│                                                                         │
│   [Stage 1: L0-L4]  ──QUIC Datagram──►  [Stage 2: L5-L7]  ───...──┐     │
│       ▲                                                           │     │
│       │ (Reverse QUIC Stream: Next Drafts [d1, d2])               │     │
│       └───────────────────────────── [Stage 24: L59-L60 + MTP] ◄──┘     │
└─────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│               Out-of-Band State, Audit & Registry Mesh                  │
│  • OrbitDB / IPFS Shard Coldstore      • Bilateral Signed Hop Receipts  │
│  • SPRT Statistical Fraud Engine       • Warm Standby KV Buddy Relays   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## License
MIT License • Personal Research & Engineering Project by [@thealidev](https://github.com/thealidev).
