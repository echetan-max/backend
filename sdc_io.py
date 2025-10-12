# sdc_io.py
import io
import os
import json
import requests
import numpy as np
from supabase import create_client

SUPABASE_URL  = os.environ["SUPABASE_URL"]
SERVICE_KEY   = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
BUCKET_IMAGES = os.environ.get("BUCKET_IMAGES", "images")
BUCKET_ART    = os.environ.get("BUCKET_ARTIFACTS", "artifacts")

sb = create_client(SUPABASE_URL, SERVICE_KEY)

def sign_get(bucket: str, path: str, ttl: int = 3600) -> bytes:
    url = sb.storage.from_(bucket).create_signed_url(path, ttl)["signedURL"]
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.content

def upload_bytes(bucket: str, path: str, data: bytes, content_type: str = "application/octet-stream"):
    sb.storage.from_(bucket).upload(path, data, {"contentType": content_type, "upsert": True})

def save_json(bucket: str, path: str, obj):
    upload_bytes(bucket, path, json.dumps(obj).encode("utf-8"), "application/json")

def save_npz(bucket: str, path: str, **kwargs):
    buf = io.BytesIO()
    np.savez_compressed(buf, **kwargs)
    upload_bytes(bucket, path, buf.getvalue(), "application/x-npz")

def load_npz_bytes(b: bytes):
    return np.load(io.BytesIO(b), allow_pickle=True)

def get_message(mid: str):
    return sb.table("messages").select("*").eq("id", mid).single().execute().data

def get_image_row(image_id: str):
    return sb.table("images").select("*").eq("id", image_id).single().execute().data

def set_status(mid: str, status: str):
    if not mid: return
    sb.table("messages").update({"status": status}).eq("id", mid).execute()

def write_encoded_manifest(mid: str, counts_path: str, meta_path: str):
    sb.table("sdc_encoded").upsert({
        "message_id": mid,
        "counts_path": counts_path,
        "meta_path": meta_path
    }).execute()

def write_decode_result(mid: str, recon_path: str, metrics: dict):
    sb.table("sdc_results").insert({
        "message_id": mid,
        "recon_path": recon_path,
        "metrics_json": metrics
    }).execute()
