# pipeline_decode.py
import io
import json
import numpy as np
from math import log10
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from sdc_io import (
    sb, sign_get, upload_bytes, load_npz_bytes, get_message, get_image_row,
    write_decode_result, set_status, BUCKET_IMAGES, BUCKET_ART
)

GRAY    = [0,1,3,2,6,7,5,4,12,13,15,14,10,11,9,8]
INVGRAY = [GRAY.index(i) for i in range(16)]

def _vec(counts: dict) -> np.ndarray:
    v = np.zeros(4, dtype=float)
    for k, c in counts.items():
        v[int(k, 2)] += float(c)
    s = v.sum()
    return v / s if s > 0 else np.full(4, 0.25)

def _mk4(C0, C1, C2, C3) -> np.ndarray:
    return np.vstack([_vec(C0), _vec(C1), _vec(C2), _vec(C3)])  # (4,4) row-stochastic

def _pinv_reg(M: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    # Tikhonov regularization for stability: (M^T M + lam I)^-1 M^T
    Mt = M.T
    return np.linalg.inv(Mt @ M + lam * np.eye(M.shape[1])) @ Mt

def _mitigated_argmax(M: np.ndarray, counts: dict) -> int:
    y = _vec(counts)        # measured
    Mpinv = _pinv_reg(M)    # robust mitigation
    x = Mpinv @ y
    x = np.clip(x, 0, 1)
    s = x.sum()
    if s > 0: x = x / s
    return int(np.argmax(x))  # 0..3

def _sum_counts_dict(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items(): out[k] = out.get(k, 0) + v
    return out

def decode_phase(message_id: str):
    msg = get_message(message_id)
    set_status(message_id, "DECODING")

    # Load artifacts
    enc = sb.table("sdc_encoded").select("*").eq("message_id", message_id).single().execute().data
    meta = json.loads(sign_get(BUCKET_ART, enc["meta_path"]).decode("utf-8"))
    arrs = load_npz_bytes(sign_get(BUCKET_ART, enc["counts_path"]))

    w, h   = int(meta["w"]), int(meta["h"])
    n_pix  = int(meta["n_pix"])
    K      = int(meta.get("K", 1))
    palette = np.array(meta["palette"], dtype=np.uint8)

    # Pairwise 4×4 calibration (paper) :contentReference[oaicite:6]{index=6}
    M_hi = _mk4(arrs["cal_hi0"].item(), arrs["cal_hi1"].item(), arrs["cal_hi2"].item(), arrs["cal_hi3"].item())
    M_lo = _mk4(arrs["cal_lo0"].item(), arrs["cal_lo1"].item(), arrs["cal_lo2"].item(), arrs["cal_lo3"].item())

    gray_vals = np.zeros(n_pix, dtype=np.uint8)

    # Data order: hi[i]_r0..rK-1, then lo[i]_r0..rK-1
    for i in range(n_pix):
        # Fuse K repeats by summing raw counts (soft combination)
        hi_sum = {}
        for r in range(K):
            hi_sum = _sum_counts_dict(hi_sum, arrs[f"data_hi_{i}_r{r}"].item())

        lo_sum = {}
        for r in range(K):
            lo_sum = _sum_counts_dict(lo_sum, arrs[f"data_lo_{i}_r{r}"].item())

        hi_hat = _mitigated_argmax(M_hi, hi_sum)  # 0..3
        lo_hat = _mitigated_argmax(M_lo, lo_sum)  # 0..3
        gray_vals[i] = ((hi_hat & 0b11) << 2) | (lo_hat & 0b11)

    # Inverse Gray → palette index → reconstruct
    inv = np.vectorize(lambda g: INVGRAY[int(g)])
    pal_idx = inv(gray_vals).reshape(h, w)

    rgb = palette[pal_idx].astype(np.uint8).reshape(h, w, 3)
    recon = Image.fromarray(rgb, "RGB")

    # Metrics vs original (same size)
    orig_row = get_image_row(msg["image_id"])
    orig = Image.open(io.BytesIO(sign_get(BUCKET_IMAGES, orig_row["original_path"]))).convert("RGB").resize((w, h), Image.NEAREST)

    A = np.array(orig, dtype=np.float32)
    B = np.array(recon, dtype=np.float32)
    mse = float(np.mean((A - B) ** 2))
    psnr = 99.0 if mse == 0 else 10 * log10((255.0 ** 2) / mse)
    ssim_val = float(ssim(A.astype(np.uint8), B.astype(np.uint8), channel_axis=2, data_range=255))

    # Save recon + metrics
    buf = io.BytesIO()
    recon.save(buf, "PNG")
    recon_path = f"recon_{message_id}.png"
    upload_bytes(BUCKET_ART, recon_path, buf.getvalue(), "image/png")

    write_decode_result(message_id, recon_path, {"psnr": psnr, "ssim": ssim_val})
    set_status(message_id, "DECODED")
