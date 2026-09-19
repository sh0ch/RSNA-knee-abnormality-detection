import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sync_code_to_jupyter import JupyterContents, _jupyter_url_from_env, _load_dotenv, _parse_server
_load_dotenv(Path(__file__).resolve().parents[1] / ".env")
base, token = _parse_server(_jupyter_url_from_env())
c = JupyterContents(base, token)

def walk(path):
    p = c.get_json(f"/api/contents/{path}?content=1")
    if p.get("type") == "directory":
        for child in p.get("content") or []:
            walk(child["path"])
    else:
        t = p.get("type")
        if path.endswith(".py") and t != "file":
            print(f"BAD: {path} type={t}")
        elif path.endswith("volume_cache.py"):
            print(f"volume_cache: type={t} size={p.get('size')}")

walk("src/rsna_knee")
