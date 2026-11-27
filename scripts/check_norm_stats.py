"""B5 check: how small do the train-split standard deviations get, per layer and pooling?

Standardization (--normalize in train.py, the depth sweep's trainnorm runs, the fusion runner) divides by the cache's
train-only std, floored at 1e-6 (collection/load_cache.py SharedCache.norm_stats). This prints, for every layer and
pooling, the min / median / max std and how many feature dims fall below 1e-6 (where the floor acts), 1e-3 and 1e-2.
Reads norm_stats/<pool>.npz with the Python standard library only, so it runs on a CARC login node without numpy.

Usage: python3 scripts/check_norm_stats.py [/scratch1/akashrav/safe_cache/v1]
See docs/b5_standardization.md for the v1 results and the decision.
"""
import ast, json, struct, sys, zipfile
from array import array
from pathlib import Path

cache = Path(sys.argv[1] if len(sys.argv) > 1 else "/scratch1/akashrav/safe_cache/v1")


def read_npy(blob):
    assert blob[:6] == b"\x93NUMPY", "not a .npy file"
    major = blob[6]
    hlen_size = 2 if major == 1 else 4
    hlen = int.from_bytes(blob[8:8 + hlen_size], "little")
    start = 8 + hlen_size
    header = ast.literal_eval(blob[start:start + hlen].decode("latin1"))
    data = blob[start + hlen:]
    dtype, shape = header["descr"], header["shape"]
    codes = {"<f4": "f", "<f8": "d", "<i8": "q", "<i4": "i"}
    if dtype not in codes:
        raise ValueError(f"unsupported dtype {dtype}")
    a = array(codes[dtype]); a.frombytes(data)
    return a, shape


names = []
manifest = cache / "manifest.json"
if manifest.exists():
    names = json.loads(manifest.read_text()).get("layer_names", [])

for npz in sorted((cache / "norm_stats").glob("*.npz")):
    with zipfile.ZipFile(npz) as z:
        std, (L, D) = read_npy(z.read("std.npy"))
        layers, _ = read_npy(z.read("layers.npy"))
        n_rows = read_npy(z.read("n_rows.npy"))[0][0] if "n_rows.npy" in z.namelist() else "?"
    print(f"\n=== pool: {npz.stem}  ({L} layers x {D} dims, train rows: {n_rows})")
    print(f"{'layer':>5} {'name':<17} {'min std':>10} {'median':>9} {'max':>10} {'<1e-6':>6} {'<1e-3':>6} {'<1e-2':>6}")
    worst = (float("inf"), None)
    for i in range(L):
        row = sorted(std[i * D:(i + 1) * D])
        mn, md, mx = row[0], row[D // 2], row[-1]
        c6 = sum(v < 1e-6 for v in row); c3 = sum(v < 1e-3 for v in row); c2 = sum(v < 1e-2 for v in row)
        lid = int(layers[i]); name = names[lid] if lid < len(names) else ""
        print(f"{lid:>5} {name:<17} {mn:>10.3g} {md:>9.3g} {mx:>10.3g} {c6:>6} {c3:>6} {c2:>6}")
        if mn < worst[0]:
            worst = (mn, lid)
    print(f"smallest std in pool '{npz.stem}': {worst[0]:.3g} at layer {worst[1]}")
