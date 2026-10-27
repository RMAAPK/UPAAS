"""
UPAAS Built-in Authentication Provider for Unsloth Studio
Exposes:
  GET  /api/auth/status
  POST /api/auth/login
  POST /api/auth/logout
Default Demo Credentials:
  Username: unsloth
  Password: 12345678
"""
from aiohttp import web
import json
import time
import hmac
import hashlib
import base64

DEMO_USER = "unsloth"
DEMO_PASS = "12345678"
JWT_SECRET = "rmaa_sovereign_studio_secret_key_2026"

def generate_jwt(username: str) -> str:
    header = base64.urlsafe_b64encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode()).decode().rstrip("=")
    now = int(time.time())
    payload = base64.urlsafe_b64encode(json.dumps({
        "sub": username,
        "exp": now + 86400 * 7,
        "iat": now
    }).encode()).decode().rstrip("=")
    sig = hmac.new(JWT_SECRET.encode(), f"{header}.{payload}".encode(), hashlib.sha256).digest()
    sig_b64 = base64.urlsafe_b64encode(sig).decode().rstrip("=")
    return f"{header}.{payload}.{sig_b64}"

async def handle_auth_status(request):
    """Answers GET /api/auth/status for Unsloth Studio React frontend."""
    return web.json_response({
        "initialized": True,
        "default_username": DEMO_USER,
        "login_mode": "single_user",
        "full_access": True,
        "requires_password_change": False,
        "bootstrap_deadline_seconds": None,
        "demo_credentials": {
            "username": DEMO_USER,
            "password": DEMO_PASS
        }
    })

async def handle_login(request):
    """Answers POST /api/auth/login."""
    try:
        data = await request.json()
        user = data.get("username", "")
        pwd = data.get("password", "")

        if (user == DEMO_USER or not user) and pwd == DEMO_PASS:
            token = generate_jwt(DEMO_USER)
            response = web.json_response({
                "access_token": token,
                "token_type": "bearer",
                "must_change_password": False,
                "role": "owner"
            })
            # Set cookie for browser session persistence
            response.set_cookie(
                "unsloth_studio_auth", token,
                max_age=86400 * 7,
                path="/",
                httponly=False,
                samesite="Lax"
            )
            return response
        else:
            return web.json_response({
                "detail": f"Invalid password. Demo password is: {DEMO_PASS}"
            }, status=401)
    except Exception as e:
        return web.json_response({"detail": str(e)}, status=400)

async def handle_logout(request):
    response = web.json_response({"status": "logged_out"})
    response.del_cookie("unsloth_studio_auth", path="/")
    return response

async def handle_identity(request):
    return web.json_response({"proof": "sovereign_identity_valid"})

async def handle_current_user(request):
    return web.json_response({
        "username": DEMO_USER,
        "account_id": "owner_account_0",
        "role": "owner",
        "permissions": ["all"]
    })
