"""
TorchJIT Supabase Registry & IPFS Transport Client
Manages model manifests, tensor CIDs, and layer pipeline routing.
"""
import os
import json
import urllib.request
import urllib.error

class TensorRegistry:
    def __init__(self, supabase_url: str = None, supabase_key: str = None):
        self.supabase_url = supabase_url or os.environ.get("SUPABASE_URL", "")
        self.supabase_key = supabase_key or os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

    def register_layer(self, model_id: str, layer_idx: int, tensor_cid: str, shape: list, dtype: str = "float32") -> dict:
        """Register a layer slice CID with the decentralized manifest."""
        payload = {
            "model_id": model_id,
            "layer_index": layer_idx,
            "tensor_cid": tensor_cid,
            "shape": shape,
            "dtype": dtype
        }
        if not self.supabase_url or not self.supabase_key:
            # Local simulated registry fallback
            return {"status": "local_mock", "entry": payload}

        req = urllib.request.Request(
            f"{self.supabase_url}/rest/v1/model_layers",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "apikey": self.supabase_key,
                "Authorization": f"Bearer {self.supabase_key}",
                "Content-Type": "application/json",
                "Prefer": "return=representation"
            },
            method="POST"
        )
        try:
            with urllib.request.urlopen(req) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            return {"error": str(e), "entry": payload}

    def get_layer_cid(self, model_id: str, layer_idx: int) -> str:
        """Fetch the IPFS / Local-JIT CID for a given layer."""
        if not self.supabase_url or not self.supabase_key:
            return f"cidv1_{model_id}_layer_{layer_idx}"

        req = urllib.request.Request(
            f"{self.supabase_url}/rest/v1/model_layers?model_id=eq.{model_id}&layer_index=eq.{layer_idx}&select=tensor_cid",
            headers={
                "apikey": self.supabase_key,
                "Authorization": f"Bearer {self.supabase_key}"
            }
        )
        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if data and len(data) > 0:
                    return data[0]["tensor_cid"]
        except Exception:
            pass
        return ""
