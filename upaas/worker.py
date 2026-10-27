"""
UPAAS Worker Node Daemon
Runs 24/7 on free PaaS instances (Render/Koyeb) with 13-minute ping keepalive.
Hardened against free-tier idling, sleep timeouts, and OOM drops.
"""
import argparse
import asyncio
import sys
import os
import urllib.request
from aiohttp import web

# Configure paths
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_dir)
if os.path.exists("/app/torchjit/python"):
    sys.path.append("/app/torchjit/python")
elif os.path.exists("/media/thealidev/RMAA HDD1/RMAAPK/TorchJIT/python"):
    sys.path.append("/media/thealidev/RMAA HDD1/RMAAPK/TorchJIT/python")

import torchjit as torch
from upaas.pipeline import DistributedPipeline

routes = web.RouteTableDef()
pipeline_node = None

@routes.get("/")
async def root(request):
    return web.json_response({
        "status": "active",
        "service": "UPAAS Layer Worker",
        "layer_idx": pipeline_node.layer_idx if pipeline_node else -1,
        "total_layers": pipeline_node.total_layers if pipeline_node else -1,
        "engine": "TorchJIT-C++20",
        "chunk_size": "1-3MB"
    })

@routes.get("/health")
async def health(request):
    return web.json_response({
        "status": "healthy",
        "layer_idx": pipeline_node.layer_idx if pipeline_node else -1
    })

@routes.post("/forward")
async def forward(request):
    try:
        data = await request.json()
        values = data.get("tensor", [])
        shape = data.get("shape", [1, len(values)])
        
        t = torch.empty(shape)
        for i, v in enumerate(values):
            t[i] = float(v)
            
        out = pipeline_node.process_activation(t)
        response_data = {
            "status": "success",
            "layer_idx": pipeline_node.layer_idx,
            "output_shape": out.shape,
            "values": [out[i] for i in range(min(10, out.numel()))] # preview
        }
        # Immediately evict intermediate layer weights to stay well under 512MB RAM
        pipeline_node.layer.evict()
        return web.json_response(response_data)
    except Exception as e:
        return web.json_response({"status": "error", "message": str(e)}, status=500)

async def keep_alive_task(port: int, interval_seconds: int = 780):
    """
    Self-ping keepalive loop: Every 13 minutes (780s), pings itself
    so Render / Koyeb free-tier web services NEVER go to sleep.
    """
    await asyncio.sleep(10)
    print(f"⚡ [UPAAS KeepAlive] Activated heartbeat loop (interval: {interval_seconds}s)")
    while True:
        try:
            url = f"http://localhost:{port}/health"
            req = urllib.request.Request(url, headers={"User-Agent": "UPAAS-Pulse/1.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                print(f"[UPAAS KeepAlive] Pulse sent to {url} - Status {resp.status}")
        except Exception as e:
            print(f"[UPAAS KeepAlive] Pulse attempt: {e}")
        await asyncio.sleep(interval_seconds)

def main():
    parser = argparse.ArgumentParser(description="UPAAS Distributed Layer Worker")
    parser.add_argument("--layer", type=int, default=int(os.environ.get("LAYER_INDEX", 0)), help="Assigned layer index")
    parser.add_argument("--total-layers", type=int, default=int(os.environ.get("TOTAL_LAYERS", 80)), help="Total pipeline depth")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 10000)), help="Service HTTP port")
    parser.add_argument("--interval", type=int, default=int(os.environ.get("KEEP_ALIVE_INTERVAL", 780)), help="Keep-alive interval in seconds")
    args = parser.parse_args()

    global pipeline_node
    chunk_dir = os.environ.get("CHUNK_CACHE_DIR", "/tmp/tjit_chunks")
    pipeline_node = DistributedPipeline(args.layer, args.total_layers, chunk_dir=chunk_dir)
    print(f"🚀 UPAAS Node Online: Layer {args.layer}/{args.total_layers} on port {args.port}")

    app = web.Application()
    app.add_routes(routes)
    
    # Background self-ping pulse
    loop = asyncio.get_event_loop()
    loop.create_task(keep_alive_task(port=args.port, interval_seconds=args.interval))
    
    web.run_app(app, host="0.0.0.0", port=args.port)

if __name__ == "__main__":
    main()
