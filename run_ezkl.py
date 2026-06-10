#!/usr/bin/env python3
"""
Run EZKL proof generation on the already-trained MLP.
Prerequisites: mlp.onnx must exist in artifacts/
"""
import json, pickle, asyncio, sys
import numpy as np
import boto3, io, pandas as pd
from pathlib import Path

OUT_DIR       = Path("/tmp/mlp_zkml/artifacts")
ONNX_PATH     = OUT_DIR / "mlp_fixed.onnx"  # fixed-shape ONNX (batch=1)
SETTINGS_PATH = OUT_DIR / "settings.json"
COMPILED_PATH = OUT_DIR / "mlp.ezkl"
SRS_PATH      = OUT_DIR / "kzg.srs"
PK_PATH       = OUT_DIR / "pk.key"
VK_PATH       = OUT_DIR / "vk.key"
WITNESS_PATH  = OUT_DIR / "witness.json"
PROOF_PATH    = OUT_DIR / "proof.json"
INPUT_PATH    = OUT_DIR / "input.json"
SCALER_PATH   = OUT_DIR / "scaler.pkl"

# Load scaler
with open(SCALER_PATH, "rb") as f:
    d = pickle.load(f)
scaler       = d["scaler"]
FEATURE_COLS = d["feature_cols"]
print(f"Scaler loaded. Features ({len(FEATURE_COLS)}): {FEATURE_COLS[:5]}...")

# Verify ONNX exists
if not ONNX_PATH.exists():
    sys.exit(f"ERROR: {ONNX_PATH} not found. Run train_and_prove.py first.")

import onnx, onnxruntime as rt
print("\nVerifying ONNX model...")
onnx_model = onnx.load(str(ONNX_PATH))
onnx.checker.check_model(onnx_model)
print("  ✓ ONNX model valid")

# Get a recent BTC feature vector from S3
print("\nBuilding sample input from latest BTC data...")
S3_BUCKET = "bitcoin-prediction-option4-production-654654488711"
s3  = boto3.client("s3")
obj = s3.get_object(Bucket=S3_BUCKET, Key="training-data/btc_hourly.parquet")
df_raw = pd.read_parquet(io.BytesIO(obj["Body"].read()))
df_raw.columns = [c.lower() for c in df_raw.columns]
if "timestamp" in df_raw.columns:
    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
    df_raw = df_raw.set_index("timestamp").sort_index()

price_col = "price" if "price" in df_raw.columns else "close"
df = df_raw.copy()
df["close"]  = df[price_col].astype(float)
if "volume" in df.columns:
    df["volume"] = df["volume"].astype(float)

df["ret_1h"]  = df["close"].pct_change(1)
df["ret_4h"]  = df["close"].pct_change(4)
df["ret_24h"] = df["close"].pct_change(24)
df["ret_7d"]  = df["close"].pct_change(24*7)
df["vol_24h"] = df["ret_1h"].rolling(24).std()
df["vol_7d"]  = df["ret_1h"].rolling(24*7).std()
sma20 = df["close"].rolling(20).mean()
std20 = df["close"].rolling(20).std()
df["bb_upper"] = (df["close"] - (sma20 + 2*std20)) / (4*std20 + 1e-9)
df["bb_pct"]   = (df["close"] - sma20) / (2*std20 + 1e-9)
df["vs_sma7"]  = df["close"] / df["close"].rolling(24*7).mean() - 1
df["vs_sma30"] = df["close"] / df["close"].rolling(24*30).mean() - 1

def rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - 100 / (1 + rs)

df["rsi14"]     = rsi(df["close"], 14) / 100.0
ema12 = df["close"].ewm(span=12).mean()
ema26 = df["close"].ewm(span=26).mean()
df["macd_norm"] = (ema12 - ema26) / (df["close"] + 1e-9)
if "volume" in df.columns:
    df["vol_ratio"] = df["volume"] / (df["volume"].rolling(24).mean() + 1e-9) - 1
else:
    df["vol_ratio"] = 0.0
if hasattr(df.index, "hour"):
    df["hour_sin"] = np.sin(2*np.pi * df.index.hour / 24)
    df["hour_cos"] = np.cos(2*np.pi * df.index.hour / 24)
    df["dow_sin"]  = np.sin(2*np.pi * df.index.dayofweek / 7)
    df["dow_cos"]  = np.cos(2*np.pi * df.index.dayofweek / 7)
else:
    df["hour_sin"] = df["hour_cos"] = df["dow_sin"] = df["dow_cos"] = 0.0
if "high" in df.columns and "low" in df.columns:
    df["hl_ratio"] = (df["high"].astype(float) - df["low"].astype(float)) / (df["close"] + 1e-9)
else:
    df["hl_ratio"] = 0.0

df = df.dropna(subset=FEATURE_COLS)
row = df[FEATURE_COLS].iloc[-1:].values.astype(np.float32)
row_scaled = scaler.transform(row)
row_scaled = np.clip(row_scaled, -5, 5).astype(np.float32)
print(f"  Latest BTC timestamp: {df.index[-1]}")
print(f"  Feature vector[:5]: {row_scaled[0][:5].tolist()}")

# Verify ONNX inference
sess = rt.InferenceSession(str(ONNX_PATH))
ort_out = sess.run(None, {"input": row_scaled})[0]
print(f"  ONNX prediction (return): {ort_out[0][0]:.6f}  ({ort_out[0][0]*100:.3f}%)")

# Write input.json
input_data = {"input_data": [row_scaled.flatten().tolist()]}
with open(INPUT_PATH, "w") as f:
    json.dump(input_data, f)
print(f"  input.json saved → {INPUT_PATH}")

# ─── EZKL Steps ───────────────────────────────────────────────────────────────
import ezkl
print(f"\nEZKL version: {ezkl.__version__}")

# Step 1: gen_settings
print("\n[1/6] gen_settings...")
run_args = ezkl.PyRunArgs()
run_args.input_visibility  = "public"
run_args.output_visibility = "public"
run_args.param_visibility  = "private"
res = ezkl.gen_settings(str(ONNX_PATH), str(SETTINGS_PATH), py_run_args=run_args)
print(f"  ✓ gen_settings = {res}")

# Step 2: calibrate_settings
print("[2/6] calibrate_settings (1-2 min)...")
res = ezkl.calibrate_settings(str(INPUT_PATH), str(ONNX_PATH), str(SETTINGS_PATH), "resources")
print(f"  ✓ calibrate = {res}")
with open(SETTINGS_PATH) as f:
    settings = json.load(f)
logrows = settings.get("run_args", {}).get("logrows", "?")
print(f"  logrows = {logrows}  (circuit size = 2^{logrows})")

# Step 3: compile_circuit
print("[3/6] compile_circuit...")
res = ezkl.compile_circuit(str(ONNX_PATH), str(COMPILED_PATH), str(SETTINGS_PATH))
print(f"  ✓ compiled → {COMPILED_PATH}")

# Step 4: get_srs — needs an active event loop in ezkl 23
print("[4/6] get_srs (fetching KZG params)...")
import asyncio

async def _get_srs():
    return ezkl.get_srs(str(SETTINGS_PATH), srs_path=str(SRS_PATH))

res = asyncio.run(_get_srs())
print(f"  ✓ kzg.srs → {SRS_PATH}  ({SRS_PATH.stat().st_size:,} bytes)")

# Step 5: setup
print("[5/6] setup (proving+verification keys)...")
res = ezkl.setup(str(COMPILED_PATH), str(VK_PATH), str(PK_PATH), srs_path=str(SRS_PATH))
pk_sz = PK_PATH.stat().st_size if PK_PATH.exists() else 0
vk_sz = VK_PATH.stat().st_size if VK_PATH.exists() else 0
print(f"  ✓ pk.key  {pk_sz:,} bytes")
print(f"  ✓ vk.key  {vk_sz:,} bytes")

# Step 6: gen_witness + prove
print("[6/6] gen_witness + prove...")
res_w = ezkl.gen_witness(str(INPUT_PATH), str(COMPILED_PATH), str(WITNESS_PATH))
print(f"  ✓ witness generated")

res_p = ezkl.prove(
    str(WITNESS_PATH), str(COMPILED_PATH), str(PK_PATH), str(PROOF_PATH),
    proof_type="single", srs_path=str(SRS_PATH),
)
pf_sz = PROOF_PATH.stat().st_size if PROOF_PATH.exists() else 0
print(f"  ✓ proof.json  {pf_sz:,} bytes")

# Verify
print("\n[Verify] Checking proof...")
res_v = ezkl.verify(str(PROOF_PATH), str(SETTINGS_PATH), str(VK_PATH), srs_path=str(SRS_PATH))
print(f"  ✓ Proof verified: {res_v}")

# ─── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("EZKL ARTIFACTS READY FOR RAINFACE UPLOAD")
print("="*60)
for path, label in [
    (VK_PATH,       "vk.key"),
    (SETTINGS_PATH, "settings.json"),
    (PROOF_PATH,    "proof.json"),
    (SRS_PATH,      "kzg.srs"),
]:
    sz = path.stat().st_size if path.exists() else 0
    ok = "✓" if path.exists() else "✗"
    print(f"  {ok}  {str(path):55s}  {sz:>10,} bytes")
print(f"\nAll files in: {OUT_DIR}")
