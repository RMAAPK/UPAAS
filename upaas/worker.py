"""
UPAAS Worker Node Daemon & Interactive Web Platform
Serves the raw, compiled official Unsloth Studio React UI at /
with complete API coverage for models, settings, system probes, and streaming inference.
"""
import argparse
import asyncio
import sys
import os
import urllib.request
from aiohttp import web

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_dir)
if os.path.exists("/app/torchjit/python"):
    sys.path.append("/app/torchjit/python")
elif os.path.exists("/media/thealidev/RMAA HDD1/RMAAPK/TorchJIT/python"):
    sys.path.append("/media/thealidev/RMAA HDD1/RMAAPK/TorchJIT/python")

import torchjit as torch
from upaas.pipeline import DistributedPipeline
from upaas.auth_router import (
    handle_auth_status,
    handle_login,
    handle_logout,
    handle_identity,
    handle_current_user
)
from upaas.studio_api import (
    api_health,
    api_system_info,
    api_models_all,
    api_models_downloaded,
    api_settings_all,
    api_engines,
    api_providers,
    api_mcp_servers,
    api_recipes,
    api_rag_status,
    api_chat_completions,
    api_accounts,
    api_accounts_me,
    api_chat_history
)

routes = web.RouteTableDef()
pipeline_node = None
DIST_DIR = os.path.join(os.path.dirname(__file__), "dist")
INDEX_HTML = os.path.join(DIST_DIR, "index.html")

# Authentication APIs
@routes.get("/api/auth/status")
async def r_auth_status(request): return await handle_auth_status(request)

@routes.post("/api/auth/login")
async def r_auth_login(request): return await handle_login(request)

@routes.post("/api/auth/logout")
async def r_auth_logout(request): return await handle_logout(request)

@routes.get("/api/auth/identity")
async def r_auth_identity(request): return await handle_identity(request)

@routes.get("/api/users/me")
async def r_users_me(request): return await handle_current_user(request)

# Accounts APIs
@routes.get("/api/accounts")
async def r_accounts(request): return await api_accounts(request)

@routes.get("/api/accounts/me")
async def r_accounts_me(request): return await api_accounts_me(request)

# System, Health & Probe APIs
@routes.get("/api/health")
async def r_api_health(request): return await api_health(request)

@routes.get("/health")
async def r_health(request): return await api_health(request)

@routes.get("/api/system")
async def r_system(request): return await api_system_info(request)

@routes.get("/api/system/info")
async def r_system_info(request): return await api_system_info(request)

@routes.get("/api/system/gpu")
async def r_system_gpu(request): return await api_system_info(request)

@routes.get("/api/engines")
async def r_engines(request): return await api_engines(request)

# Models APIs
@routes.get("/api/models")
async def r_models(request): return await api_models_all(request)

@routes.get("/api/models/downloaded")
async def r_models_down(request): return await api_models_downloaded(request)

# Settings & Preferences APIs
@routes.get("/api/settings")
async def r_settings(request): return await api_settings_all(request)

@routes.get("/api/settings/models")
async def r_settings_models(request): return await api_settings_all(request)

@routes.get("/api/preferences")
async def r_preferences(request): return await api_settings_all(request)

# Chat, MCP, RAG APIs
@routes.get("/api/chat-history")
async def r_chat_history(request): return await api_chat_history(request)

@routes.get("/api/threads")
async def r_threads(request): return await api_chat_history(request)

@routes.post("/v1/chat/completions")
async def r_chat_v1(request): return await api_chat_completions(request)

@routes.post("/api/inference/chat")
async def r_chat_inf(request): return await api_chat_completions(request)

@routes.get("/api/providers")
async def r_providers(request): return await api_providers(request)

@routes.get("/api/mcp/servers")
async def r_mcp(request): return await api_mcp_servers(request)

@routes.get("/api/recipes")
async def r_recipes(request): return await api_recipes(request)

@routes.get("/api/rag/status")
async def r_rag(request): return await api_rag_status(request)

@routes.get("/api/status")
async def status(request):
    return web.json_response({
        "status": "active",
        "service": "UPAAS Layer Worker",
        "layer_idx": pipeline_node.layer_idx if pipeline_node else -1,
        "total_layers": pipeline_node.total_layers if pipeline_node else -1,
        "engine": "TorchJIT-C++20",
        "chunk_size": "1-3MB",
        "platform": "Unsloth Over Platform As A Service"
    })

@routes.post("/forward")
async def forward(request):
    try:
        data = await request.json()
        values = data.get("tensor", [1.0, 1.0, 1.0, 1.0])
        shape = data.get("shape", [1, len(values)])
        t = torch.empty(shape)
        for i, v in enumerate(values):
            t[i] = float(v)
        out = pipeline_node.process_activation(t)
        response_data = {
            "status": "success",
            "layer_idx": pipeline_node.layer_idx,
            "output_shape": out.shape,
            "values": [out[i] for i in range(min(10, out.numel()))]
        }
        pipeline_node.layer.evict()
        return web.json_response(response_data)
    except Exception as e:
        return web.json_response({"status": "error", "message": str(e)}, status=500)

# Serve root index.html explicitly
@routes.get("/")
async def root_index(request):
    if os.path.exists(INDEX_HTML):
        return web.FileResponse(INDEX_HTML)
    return web.json_response({"status": "active", "service": "UPAAS Unsloth Studio Node"})

# Single Page App (SPA) fallback handler
@routes.get("/{tail:.*}")
async def spa_fallback(request):
    tail = request.match_info["tail"]
    target_path = os.path.join(DIST_DIR, tail)
    if os.path.isfile(target_path):
        return web.FileResponse(target_path)
    # SPA fallback: Return index.html for all client-side routes (e.g. /chat, /models)
    if os.path.exists(INDEX_HTML):
        return web.FileResponse(INDEX_HTML)
    return web.Response(status=404, text="Not Found")

async def keep_alive_task(port: int, interval_seconds: int = 780):
    await asyncio.sleep(10)
    print(f"⚡ [UPAAS KeepAlive] Pulse active on port {port} every {interval_seconds}s")
    while True:
        try:
            url = f"http://localhost:{port}/health"
            req = urllib.request.Request(url, headers={"User-Agent": "UPAAS-Pulse/1.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                pass
        except Exception:
            pass
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
    print(f"🚀 UPAAS Official Unsloth Studio Online on port {args.port}")

    app = web.Application()
    app.add_routes(routes)
    
    loop = asyncio.get_event_loop()
    loop.create_task(keep_alive_task(port=args.port, interval_seconds=args.interval))
    
    web.run_app(app, host="0.0.0.0", port=args.port)

if __name__ == "__main__":
    main()
