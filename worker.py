# worker.py
import os
import time
import json
import traceback
from supabase import create_client
from sdc_io import set_status
from pipeline_encode import run_encode_phase
from pipeline_decode import decode_phase

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

def pop_job():
    res = sb.table("job_queue").select("*").eq("status", "ready").order("created_at", desc=False).limit(1).execute().data
    if not res:
        return None
    job = res[0]
    sb.table("job_queue").update({"status": "taken"}).eq("id", job["id"]).execute()
    return job

print("Worker started; polling for jobs...")
while True:
    job = pop_job()
    if not job:
        time.sleep(2)
        continue
    try:
        name = job["name"]
        payload = job["payload"] if isinstance(job["payload"], dict) else json.loads(job["payload"])
        mid = payload.get("message_id", "")
        print(f"[{job['id']}] → {name} → message {mid}")

        if name == "sdc.encode":
            run_encode_phase(mid)
        elif name == "sdc.decode":
            decode_phase(mid)
        else:
            raise RuntimeError(f"Unknown job: {name}")

        sb.table("job_queue").update({"status": "done"}).eq("id", job["id"]).execute()
        print(f"[{job['id']}] ✓ done")
    except Exception:
        traceback.print_exc()
        set_status(payload.get("message_id", ""), "FAILED")
        sb.table("job_queue").update({"status": "failed"}).eq("id", job["id"]).execute()
