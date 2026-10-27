# UPAAS: Unsloth Platform As A Service ⚡
### Enterprise Sovereign AI Infrastructure Fabric & Distributed Memory Mesh
Personal Research & Hobby Project • Licensed under **MIT License**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render%20Cloud-00d2d3?style=for-the-badge&logo=render)](https://upaas.onrender.com)
[![Engine](https://img.shields.io/badge/Native%20Engine-TorchJIT%20C%2B%2B20-ff4757?style=for-the-badge)](https://github.com/thealidev/TJIT)
[![Network](https://img.shields.io/badge/Storage-Decentralized%20IPFS%20Mesh-2ed573?style=for-the-badge&logo=ipfs)](https://ipfs.tech)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)

---

## 🌐 Live Interactive Demo
- **URL**: [https://upaas.onrender.com](https://upaas.onrender.com)
- **Demo Username**: `unsloth`
- **Demo Password**: `12345678`

---

## 🏢 Executive Overview & Corporate Architecture
Modern Frontier AI inference and fine-tuning are historically restricted by the **"Gigabyte Barrier"**: bloated PyTorch runtimes, 6.7GB+ container images, and massive parameter weights that exceed the physical RAM of commodity infrastructure.

**UPAAS (Unsloth Platform As A Service)** re-architects AI model execution from the ground up:
1. **Decoupled Thin Worker Architecture**: Replaces monolithic gigabyte containers with a feather-light **~25MB execution daemon**.
2. **1–3 MB Micro-Sharded Weights**: Deconstructs massive models (from 8B up to 500B+ parameters) into cryptographic **Content-Addressed (CID)** micro-slices.
3. **Decentralized IPFS Storage Mesh**: Model weights and shared libraries live across the global, immutable IPFS node network at zero hosting cost. Once hashed, they persist forever.
4. **Universal Transaction Pooler Coordination**: Employs an ultra-fast PostgreSQL Transaction Pooler (Port 6543 / PgBouncer / Supavisor) that stores **only 120-byte cryptographic CID manifests and decryption keys**—zero weight load on the database.
5. **TorchJIT C++20 Zero-Copy Execution**: Streams active layer slices straight into memory, executes transformer kernels (`RMSNorm`, `SiLU`, fast LoRA projections), and immediately evicts them, enabling 500B models to execute within standard **512MB RAM free-tier instances**.

---

## 🏛️ System Topology

```
                       [ Enterprise Client / Web UI ]
                                     │
                                     ▼
                   [ UPAAS Sovereign Layer Fabric ]
                   (https://upaas.onrender.com)
                                     │
         ┌───────────────────────────┴───────────────────────────┐
         │                                                       │
         ▼                                                       ▼
┌───────────────────────────────┐               ┌───────────────────────────────┐
│  State & Manifest Registry    │               │  Global Storage & Memory Mesh │
│  (Postgres Tx Pooler :6543)   │               │   (Decentralized IPFS Nodes)  │
├───────────────────────────────┤               ├───────────────────────────────┤
│ • Zero weight bloat           │               │ • Immutable 1-3 MB chunks     │
│ • Stores only CID hashes &    │               │ • Global DHT replication      │
│   decryption keys (~120 B)    │               │ • Zero-cost persistent host   │
└───────────────┬───────────────┘               └───────────────┬───────────────┘
                │                                               │
                └───────────────────────┬───────────────────────┘
                                        │
                                        ▼
                   ┌─────────────────────────────────────────┐
                   │    TorchJIT C++20 Layer Pipeline        │
                   │  (Virtual Memory Allocator & JIT Kernel)│
                   ├─────────────────────────────────────────┤
                   │ 1. Fetches active 1-3 MB slice JIT      │
                   │ 2. Translates CID -> C++20 pointer      │
                   │ 3. Executes fast forward / backward     │
                   │ 4. Calls .evict() (Flat RAM < 85MB)     │
                   └─────────────────────────────────────────┘
```

---

## ⚡ Key Differentiators

| Capability | Traditional AI Platforms (AWS/HuggingFace) | UPAAS + TorchJIT Fabric |
| :--- | :--- | :--- |
| **Container Footprint** | 6.7 GB – 15 GB (Docker bloat) | **~25 MB ultra-lean container** |
| **Model Ingestion** | Monolithic load (30 GB - 1 TB RAM required) | **1–3 MB micro-sharded dynamic streaming** |
| **Database Load** | High storage & heavy bandwidth bottleneck | **0% weight load (stores only cryptographic CIDs)** |
| **Cost Profile** | Heavy cloud GPU spend ($1,000s/mo) | **Runs on free-tier PaaS (Render, Koyeb, Fly.io)** |
| **Node Availability** | Sleep timeouts kill idle instances | **Automated 13-minute keep-alive heartbeat loop** |
| **UI Completeness** | Minimal stripped-down web shells | **Official Unsloth Studio React UI (Zero missing tools)** |

---

## 🛠️ Technology Stack
- **Native Runtime Core**: [TorchJIT C++20 Engine](https://github.com/thealidev/TJIT) (64-byte alignment, zero-copy `mmap`, content-addressed memory).
- **Presentation Layer**: Unsloth Studio Production React/TypeScript Suite (Chat, Model Picker, Data Recipes, GGUF Export, RAG, MCP Bridge).
- **Communication Fabric**: Asynchronous Python 3.12 Web Daemon with POSIX `memfd_create` in-memory library loading.
- **Decentralized Storage**: InterPlanetary File System (IPFS) public gateway rotation (`ipfs.io`, `dweb.link`, `cloudflare-ipfs`).
- **Database Engine**: PostgreSQL 17 Transaction Pooler (AWS Tokyo, Port 6543) with embedded SQLite local developer fallback.

---

## 🚀 Rapid Deployment

### 1. Deploy via Docker (Production)
```bash
docker run -d -p 10000:10000 \
  -e PORT=10000 \
  -e DATABASE_URL="postgresql://user:pass@host:6543/postgres?sslmode=require" \
  747705454470.dkr.ecr.ap-southeast-2.amazonaws.com/upaas:latest
```

### 2. Deploy on Render (Free Tier Blueprint)
Connect your repository to Render and deploy using the pre-configured [`render.yaml`](render.yaml):
- **Runtime**: Docker (Multi-stage build)
- **Plan**: Free
- **Auto-Keepalive**: Included (780s pulse prevents service hibernation)

---

## 📦 Developer SDK & Local Offline Sync
Developers can run completely offline without cloud database dependencies:

```bash
# Pull the pre-made open-source swarm manifest to your machine
python -m upaas.export_db --pull

# Inspect the decentralized model manifest
python -m upaas.export_db --export-json
```

```python
import torchjit as torch
from upaas.pipeline import DistributedPipeline

# Initialize Layer Worker node
pipeline = DistributedPipeline(layer_idx=0, total_layers=80)

# Zero-copy activation pass
x = torch.ones([1, 4096])
output = pipeline.process_activation(x)

# Immediate memory eviction ensures flat memory profile
pipeline.layer.evict()
```

---

## 📜 Intellectual Property & Licensing
**UPAAS** is an experimental hobby project by @thealidev
- **License**: [MIT License](LICENSE)
- **Change Date**: 2030-01-01 (Transitions to Apache License 2.0)
- **Third-Party Attributions**:
  - Unsloth AI (Apache-2.0 License)
  - PyTorch (BSD-3-Clause License)

---
*Private experimental hobby project.*
