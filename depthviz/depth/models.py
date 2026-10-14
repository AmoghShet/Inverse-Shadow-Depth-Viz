"""Pinned depth models: download URL, sha256, file size and input spec. `fetch` downloads and verifies.

The runtime never downloads. A missing model is an error that names the command:
    python -m depthviz fetch-models [--model dav2_small_q8]

All models output relative INVERSE depth (larger = nearer), with an unknown per-frame scale and shift.
Depth Anything V2 accepts any input whose sides are multiples of 14 (dynamic shape, verified at M0);
MiDaS v2.1 small is fixed at 256x256. Both use ImageNet mean/std on RGB in 0..1, NCHW float32.
The fp16 / q4f16 exports fail ORT's full graph optimisation on the CPU EP (a SimplifiedLayerNorm
fusion bug) and are loaded with basic optimisation (`opt="basic"`).
"""
import hashlib
import os
import sys
import time
import urllib.request

from .. import MODEL_DIR

_HF = "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/"

MODELS = {
    "midas_small": {
        "file": "midas_v21_small_256.onnx",
        "url": "https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx",
        "sha256": "2d8c6cb8f415229daf1eb041024208e2608c9f98e17c81cc7c6ecb449c56fd58",
        "bytes": 66764249,
        "kind": "midas",
        "fixed": (256, 256),
        "opt": "all",
        "label": "MiDaS v2.1 small (fp32, 256x256)",
    },
    "dav2_small": {
        "file": "dav2_small_model.onnx",
        "url": _HF + "model.onnx",
        "sha256": "afb6a5c28f3b6bf1618c6e43f02073ef9dfdc70e937502d51603e57b0a1df10c",
        "bytes": 99060839,
        "kind": "dav2",
        "fixed": None,
        "opt": "all",
        "label": "Depth Anything V2 small (fp32)",
    },
    "dav2_small_fp16": {
        "file": "dav2_small_model_fp16.onnx",
        "url": _HF + "model_fp16.onnx",
        "sha256": "2df6223f206b5164e21f664ace61dabeb9bb6a49b8b5a3e00510b4807d0f5b04",
        "bytes": 49642442,
        "kind": "dav2",
        "fixed": None,
        "opt": "basic",
        "label": "Depth Anything V2 small (fp16 weights)",
    },
    "dav2_small_q8": {
        "file": "dav2_small_model_quantized.onnx",
        "url": _HF + "model_quantized.onnx",
        "sha256": "fcf51f1b230362b28690bb9d1809bf0431f29cad20534e3f589bd7285547f20d",
        "bytes": 27258801,
        "kind": "dav2",
        "fixed": None,
        "opt": "all",
        "label": "Depth Anything V2 small (int8 dynamic quantised)",
    },
    "dav2_small_q4f16": {
        "file": "dav2_small_model_q4f16.onnx",
        "url": _HF + "model_q4f16.onnx",
        "sha256": "eca72971aea64216d767c70c534160de53b5435b588d362bac6dbd5a73f9bf1e",
        "bytes": 19126267,
        "kind": "dav2",
        "fixed": None,
        "opt": "basic",
        "label": "Depth Anything V2 small (4-bit weights, fp16)",
    },
}

# Depth Anything V2 input sizes tried at M0 (4:3, multiples of 14): (w, h)
DAV2_SIZES = [(252, 196), (336, 252), (420, 308)]
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def model_path(model_id):
    return os.path.join(MODEL_DIR, MODELS[model_id]["file"])


def input_size(model_id, size=None):
    """(w, h) the model is fed. `size` is the longer side request for DA-V2 (252, 336, 420)."""
    spec = MODELS[model_id]
    if spec["fixed"]:
        return spec["fixed"]
    if size is None:
        return DAV2_SIZES[1]
    for w, h in DAV2_SIZES:
        if w == int(size):
            return (w, h)
    w = max(14, int(round(int(size) / 14.0)) * 14)
    h = max(14, int(round(w * 0.75 / 14.0)) * 14)
    return (w, h)


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def ensure(model_id):
    """Path of a present model, or a clear error naming the fetch command."""
    if model_id not in MODELS:
        raise SystemExit(f"unknown depth model {model_id!r}; known: {', '.join(MODELS)}")
    p = model_path(model_id)
    if not os.path.isfile(p):
        raise SystemExit(f"depth model {model_id} is not downloaded ({p}).\n"
                         f"Run:  python -m depthviz fetch-models --model {model_id}")
    return p


def fetch(model_ids=None, verify_existing=True):
    """Download (if missing) and sha256-verify models into models/depth/. Returns {id: status}."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    out = {}
    for mid in (model_ids or list(MODELS)):
        spec = MODELS[mid]
        p = model_path(mid)
        if os.path.isfile(p):
            if not verify_existing or sha256(p) == spec["sha256"]:
                out[mid] = "present"
                print(f"{mid:>18}: present ({p})")
                continue
            print(f"{mid:>18}: sha256 mismatch, downloading again")
        tmp = p + ".part"
        t0 = time.perf_counter()
        print(f"{mid:>18}: downloading {spec['url']} ({spec['bytes'] / 1e6:.0f} MB)")
        req = urllib.request.Request(spec["url"], headers={"User-Agent": "depthviz"})
        with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as fh:
            got = 0
            last = time.perf_counter()
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                fh.write(b)
                got += len(b)
                if time.perf_counter() - last > 2.0:
                    last = time.perf_counter()
                    sys.stdout.write(f"\r{'':>20}{got / 1e6:6.1f} / {spec['bytes'] / 1e6:.1f} MB")
                    sys.stdout.flush()
        digest = sha256(tmp)
        if digest != spec["sha256"]:
            os.remove(tmp)
            raise SystemExit(f"\n{mid}: sha256 mismatch ({digest}), file removed")
        os.replace(tmp, p)
        out[mid] = "downloaded"
        print(f"\r{mid:>18}: ok in {time.perf_counter() - t0:.1f} s, sha256 verified")
    return out
