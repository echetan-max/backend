# pipeline_encode.py
import io
import os
import numpy as np
from typing import List, Tuple
from PIL import Image
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from sdc_io import (
    sign_get, save_json, save_npz, get_message, get_image_row,
    set_status, write_encoded_manifest, BUCKET_IMAGES, BUCKET_ART
)

# Practical caps for IBM trial/backends
MAX_W, MAX_H = 16,16  # max image dimensions
DEFAULT_SHOTS = 2048
DEFAULT_K = int(os.environ.get("SDC_REDUNDANCY_K", "5"))  # repeats per symbol (fusion later)

# 16-color Gray reindex (as in paper QPM + Gray) 
GRAY = [0,1,3,2,6,7,5,4,12,13,15,14,10,11,9,8]

def qpm16_gray(img: Image.Image):
    q = img.convert("P", palette=Image.Palette.ADAPTIVE, colors=16)
    pal = np.array(q.getpalette()[:48], dtype=np.uint8).reshape(16,3)
    idx = np.array(q, dtype=np.uint8)
    gidx = np.vectorize(lambda x: GRAY[int(x)])(idx).astype(np.uint8)
    return gidx, pal

def split_symbols(gidx: np.ndarray) -> Tuple[list[int], list[int]]:
    hi = ((gidx >> 2) & 0b11).astype(np.uint8).flatten().tolist()
    lo = (gidx & 0b11).astype(np.uint8).flatten().tolist()
    return hi, lo

# -------- Superdense coding (2 qubits, 2 classical bits) per rail --------

def sdc_prepare_bell_pair() -> QuantumCircuit:
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    return qc

def sdc_encode_bits(qc: QuantumCircuit, b2: int):
    # 00:I, 01:Z, 10:X, 11:XZ
    b2 &= 0b11
    if b2 == 0b01:
        qc.z(0)
    elif b2 == 0b10:
        qc.x(0)
    elif b2 == 0b11:
        qc.x(0); qc.z(0)

def sdc_bell_measure(qc: QuantumCircuit):
    qc.cx(0,1)
    qc.h(0)
    qc.measure([0,1], [0,1])

def make_sdc_circuit(message_bits: int, name: str) -> QuantumCircuit:
    qc = sdc_prepare_bell_pair()
    sdc_encode_bits(qc, message_bits)
    sdc_bell_measure(qc)
    qc.name = name
    return qc

def build_pairwise_calib_circuits() -> Tuple[List[QuantumCircuit], List[QuantumCircuit]]:
    states = [0b00, 0b01, 0b10, 0b11]
    cal_hi = [make_sdc_circuit(s, name=f"cal_hi_{s:02b}") for s in states]
    cal_lo = [make_sdc_circuit(s, name=f"cal_lo_{s:02b}") for s in states]
    return cal_hi, cal_lo

def build_data_circuits(sym_hi: list[int], sym_lo: list[int], K: int) -> List[QuantumCircuit]:
    # For each symbol build K repeats (robust fusion later)
    hi_circs = [make_sdc_circuit(b, name=f"data_hi_{i}_r{r}")
                for i, b in enumerate(sym_hi) for r in range(K)]
    lo_circs = [make_sdc_circuit(b, name=f"data_lo_{i}_r{r}")
                for i, b in enumerate(sym_lo) for r in range(K)]
    return hi_circs + lo_circs

def qd2counts(qd, shots: int) -> dict:
    out = {}
    for k, p in qd.items():
        key = format(k if isinstance(k, int) else int(k), "02b")
        c = int(round(float(p) * shots))
        if c > 0:
            out[key] = out.get(key, 0) + c
    return out

def run_encode_phase(message_id: str):
    msg = get_message(message_id)
    backend = (msg.get("backend") or os.environ.get("IBM_BACKEND") or "ibm_torino")
    shots = int(msg.get("shots") or DEFAULT_SHOTS)
    K = int(msg.get("k") or DEFAULT_K)

    set_status(message_id, "ENCODING")

    # IBM Runtime
    svc = QiskitRuntimeService(channel="ibm_cloud", token=os.environ["IBM_QUANTUM_API_KEY"])
    sampler = Sampler(service=svc, backend=backend)

    # Load + clamp image
    img_row = get_image_row(msg["image_id"])
    img = Image.open(io.BytesIO(sign_get(BUCKET_IMAGES, img_row["original_path"]))).convert("RGB")
    img = img.resize((min(img.width, MAX_W), min(img.height, MAX_H)), Image.NEAREST)

    # QPM + Gray + split to rails (paper’s front-end compression) :contentReference[oaicite:3]{index=3}
    gidx, palette = qpm16_gray(img)
    sym_hi, sym_lo = split_symbols(gidx)
    n_pix = len(sym_hi)

    # Calibration (pairwise 4×4 per rail, as in paper) :contentReference[oaicite:4]{index=4}
    cal_hi, cal_lo = build_pairwise_calib_circuits()

    # Data (K redundancy)
    data_circuits = build_data_circuits(sym_hi, sym_lo, K)

    # Run on hardware
    cal_res  = sampler.run(cal_hi + cal_lo, shots=shots).result().quasi_dists
    data_res = sampler.run(data_circuits, shots=shots).result().quasi_dists

    cal_counts = [qd2counts(d, shots) for d in cal_res]
    data_counts = [qd2counts(d, shots) for d in data_res]

    # Persist artifacts
    counts_path = f"counts_{message_id}.npz"
    meta_path   = f"meta_{message_id}.json"

    # Save calibration + all repeated data counts
    payload = {
        "cal_hi0": cal_counts[0], "cal_hi1": cal_counts[1], "cal_hi2": cal_counts[2], "cal_hi3": cal_counts[3],
        "cal_lo0": cal_counts[4], "cal_lo1": cal_counts[5], "cal_lo2": cal_counts[6], "cal_lo3": cal_counts[7],
    }
    # data order: [hi[i]_r, ... for i, r] + [lo[i]_r, ...]
    for i in range(n_pix):
        for r in range(K):
            payload[f"data_hi_{i}_r{r}"] = data_counts[i*K + r]
    base = n_pix*K
    for i in range(n_pix):
        for r in range(K):
            payload[f"data_lo_{i}_r{r}"] = data_counts[base + i*K + r]

    save_npz(BUCKET_ART, counts_path, **payload)

    save_json(BUCKET_ART, meta_path, {
        "w": img.width, "h": img.height,
        "palette": palette.tolist(),
        "n_pix": n_pix,
        "shots": shots,
        "backend": backend,
        "K": K,
        "order": "hi[i]_r[0..K-1], then lo[i]_r[0..K-1]"
    })

    write_encoded_manifest(message_id, counts_path, meta_path)
    set_status(message_id, "AWAITING_DECODE")
