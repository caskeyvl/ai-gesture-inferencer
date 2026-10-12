import argparse
import sys
import glob                 # rglob()
import numpy as np

from pathlib import Path

p = argparse.ArgumentParser(description="Augment accelerometer gesture CSVs.")
p.add_argument("--input", nargs="+", default=None, help="CSV file(s) or glob(s). Default: every CSV under data/raw")
p.add_argument("--output", default="data/augmented", help="path to output augmented CSVs")
p.add_argument("--methods", nargs="+",
               choices=["jitter", "scale", "shift", "rotate"],
               default=["jitter", "scale", "shift", "rotate"])

SEED = 0

def resolve_inputs(patterns):
    files = set()
    for pat in patterns:
        matches = glob.glob(pat, recursive=True)
        if not matches:
            sys.exit(f"error: no files match '{pat}'")
        for m in matches:
            p = Path(m)
            if p.is_dir(): 
                sys.exit(f"error: --input expects CSV files, received directory: {p}")
            if p.suffix.lower() != ".csv":
                sys.exit(f"error: --input expects CSV files, got: {p}")
            files.add(p)
    return sorted(files)

def jitter(data, rng):
    out = data.copy()
    sigma = rng.uniform(0.02, 0.1)
    out[:, 1:] += rng.normal(0, sigma, out[:, 1:].shape)
    return out

def scale(data, rng):
    out = data.copy()
    out[:, 1:] *= rng.uniform(0.8, 1.2)
    return out

def shift(data, rng):
    out = data.copy()
    n = len(out)
    max_k = n // 10                             # up to +/- 10% of recording
    k = int(rng.integers(-max_k, max_k + 1))
    if k == 0:
        return out
    vals = out[:, 1:]
    shifted = np.roll(vals, k, axis=0)
    if k > 0:
        shifted[:k] = vals[0]                   # pad beginning w/ first sample
    else:
        shifted[k:] = vals[-1]                  # pad end w/ last sample
    out[:, 1:] = shifted
    return out

def rotate(data, rng, max_deg=10):
    out = data.copy()
    ax, ay, az = np.radians(rng.uniform(-max_deg, max_deg, 3))
    Rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    Ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    Rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    out[:, 1:] = out[:, 1:] @ R.T
    return out

METHODS = {
    "jitter": jitter,
    "scale": scale,
    "shift": shift,
    "rotate": rotate
}

def main():
    args = p.parse_args()
    rng = np.random.default_rng(SEED)

    if args.input:
        files = resolve_inputs(args.input)
    else:
        files = sorted(Path("data/raw").rglob("*.csv"))
    
    if not files: 
        sys.exit("error: no CSV files found at /data/raw; run from repo root or pass --input arg)")
    for f in files:
        data = np.loadtxt(f, delimiter=',', skiprows=1)
        for m in args.methods:
            out_data = METHODS[m](data, rng)

            out_path = Path(args.output) / f.parent.name / f"{f.stem}_{m}.csv"
            out_path.parent.mkdir(parents=True, exist_ok=True)

            np.savetxt(out_path, out_data, delimiter=",", fmt="%.3f",
                       header="timestamp,accel_x,accel_y,accel_z", comments="")

if __name__ == "__main__":
    main()
