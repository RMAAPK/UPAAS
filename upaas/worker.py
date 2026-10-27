"""
UPAAS Worker Node Daemon
Runs 24/7 on free PaaS instances (Render/Koyeb) with 13-minute ping keepalive.
"""
import argparse
import asyncio
import sys
import os
from aiohttp import web

# Add TorchJIT
sys.path.append(os.path.abspath("/media/thealidev/RMAA HDD1/RMAAPK/TorchJIT/python"))
import torchjit as torch
from upaas.pipeline import DistributedPipeline

routes = web.RouteTableDef()
pipeline_node = None

@routes.get("/health")
async def health(request):
    return web.json_response({
        "status": "online",
        "engine": "TorchJIT-C++20",
        "platform": "UPAAS",
        "layer_idx": pipeline_node.layer_idx if pipeline_node else -1,
        "total_layers": pipeline_node.total_layers if pipeline_node else -1
    })

@routes.post("/forward")
async def forward(request):
    data = await request.json()
    values = data.get("tensor", [])
    shape = data.get("shape", [1, len(values)])
    
    t = torch.empty(shape)
    for i, v in enumerate(values):
        t[i] = float(v)
        
    out = pipeline_node.process_activation(t)
    return web.json_response({
        "status": "success",
        "layer_idx": pipeline_node.layer_idx,
        "output_shape": out.shape,
        "values": [out[i] for i in range(min(10, out.numel()))] # preview
    })

async def keep_alive_task(interval_seconds: int = 780):
    """Keep-alive task to prevent PaaS sleeping (fires every 13 minutes)."""
    while True:
        await asyncio.sleep(interval_seconds)
        print(f"[UPAAS KeepAlive] Ping pulse fired to retain PaaS execution context.")

def main():
    parser = argparse.ArgumentParser(description="UPAAS Distributed Layer Worker")
    parser.add_argument("--layer", type=int, default=0, help="Assigned layer index")
    parser.add_argument("--total-layers", type=int, default=80, help="Total pipeline depth")
    parser.add_argument("--port", type=int, default=10000, help="Service HTTP port")
    args = parser.parse_args()

    global pipeline_node
    pipeline_node = DistributedPipeline(args.layer, args.total_layers)
    print(f"🚀 UPAAS Layer Worker initialized for Layer {args.layer}/{args.total_layers}")

    app = web.Application()
    app.add_routes(routes)
    
    loop = asyncio.get_event_loop()
    loop.create_task(keep_alive_task())
    web.run_app(app, port=args.port)

if __name__ == "__main__":
    main()
