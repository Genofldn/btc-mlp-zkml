#!/usr/bin/env python3
"""
MLP Proxy Model for ZKML / EZKL Verification — Rainface Competition
Architecture: Dense(64,relu) → Dense(32,relu) → Dense(1,linear)   ≈4,000 params
Target      : return_24h (24-hour % return, not absolute price)
EZKL Output : vk.key  settings.json  proof.json  kzg.srs

Usage:
    python3 train_and_prove.py
"""

import os, json, pickle, asyncio
import numpy as np
import pandas as pd
from pathlib import Path

# ─── Config ────────────────────────────────────────────────────────────────────
S3_BUCKET   = "bitcoin-prediction-option4-production-654654488711"
S3_KEY      = "training-data/btc_hourly.parquet"
OUT_DIR     = Path("/tmp/mlp_zkml/artifacts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ONNX_PATH       = OUT_DIR / "mlp.onnx"
SETTINGS_PATH   = OUT_DIR / "settings.json"
COMPILED_PATH   = OUT_DIR / "mlp.ezkl"
SRS_PATH        = OUT_DIR / "kzg.srs"
PK_PATH         = OUT_DIR / "pk.key"
VK_PATH         = OUT_DIR / "vk.key"
WITNESS_PATH    = OUT_DIR / "witness.json"
PROOF_PATH      = OUT_DIR / "proof.json"
INPUT_PATH      = OUT_DIR / "input.json"
CAL_DATA_PATH   = OUT_DIR / "cal_data.json"
SCALER_PATH     = OUT_DIR / "scaler.pkl"

# ─── 1. Load BTC data ──────────────────────────────────────────────────────────
print("Loading BTC data from S3...")
import boto3, io
s3     = boto3.client("s3")
obj    = s3.get_object(Bucket=S3_BUCKET, Key=S3_KEY)
df_raw = pd.read_parquet(io.BytesIO(obj["Body"].read()))

print(f"  Rows: {len(df_raw):,}  |  Cols: {list(df_raw.columns)}")
print(f"  Range: {df_raw.index[0]} → {df_raw.index[-1]}")

# ─── 2. Feature Engineering ────────────────────────────────────────────────────
print("Engineering features...")

df = df_raw.copy()
df.columns = [c.lower() for c in df.columns]

# Set timestamp as datetime index
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    print(f"  Date range: {df.index[0]} → {df.index[-1]}")

# Map price column — parquet uses 'price' instead of 'close'
price_col = "price" if "price" in df.columns else ("close" if "close" in df.columns else df.select_dtypes(include="number").columns[0])
vol_col   = "volume" if "volume" in df.columns else None

df["close"]  = df[price_col].astype(float)
if vol_col:
    df["volume"] = df[vol_col].astype(float)

# Returns at various lags
df["ret_1h"]  = df["close"].pct_change(1)
df["ret_4h"]  = df["close"].pct_change(4)
df["ret_24h"] = df["close"].pct_change(24)
df["ret_7d"]  = df["close"].pct_change(24*7)

# Target: forward 24h return
df["target"] = df["close"].pct_change(24).shift(-24)

# Rolling stats
df["vol_24h"] = df["ret_1h"].rolling(24).std()
df["vol_7d"]  = df["ret_1h"].rolling(24*7).std()

# Bollinger Bands (20-period)
sma20 = df["close"].rolling(20).mean()
std20 = df["close"].rolling(20).std()
df["bb_upper"] = (df["close"] - (sma20 + 2*std20)) / (4*std20 + 1e-9)
df["bb_pct"]   = (df["close"] - sma20) / (2*std20 + 1e-9)

# SMA ratios
df["vs_sma7"]  = df["close"] / df["close"].rolling(24*7).mean() - 1
df["vs_sma30"] = df["close"] / df["close"].rolling(24*30).mean() - 1

# RSI (14-period)
def rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - 100 / (1 + rs)
df["rsi14"] = rsi(df["close"], 14) / 100.0  # normalise to [0,1]

# MACD signal
ema12 = df["close"].ewm(span=12).mean()
ema26 = df["close"].ewm(span=26).mean()
df["macd_norm"] = (ema12 - ema26) / (df["close"] + 1e-9)

# Volume ratio (vs 24h rolling avg)
if vol_col:
    df["vol_ratio"] = df["volume"] / (df["volume"].rolling(24).mean() + 1e-9) - 1
else:
    df["vol_ratio"] = 0.0

# Time cyclicals (hour, day-of-week)
if hasattr(df.index, "hour"):
    df["hour_sin"]    = np.sin(2*np.pi * df.index.hour / 24)
    df["hour_cos"]    = np.cos(2*np.pi * df.index.hour / 24)
    df["dow_sin"]     = np.sin(2*np.pi * df.index.dayofweek / 7)
    df["dow_cos"]     = np.cos(2*np.pi * df.index.dayofweek / 7)
else:
    df["hour_sin"] = df["hour_cos"] = df["dow_sin"] = df["dow_cos"] = 0.0

# High/Low range ratio
if "high" in df.columns and "low" in df.columns:
    df["hl_ratio"] = (df["high"].astype(float) - df["low"].astype(float)) / (df["close"] + 1e-9)
else:
    df["hl_ratio"] = 0.0

FEATURE_COLS = [
    "ret_1h", "ret_4h", "ret_24h", "ret_7d",
    "vol_24h", "vol_7d",
    "bb_upper", "bb_pct",
    "vs_sma7", "vs_sma30",
    "rsi14", "macd_norm",
    "vol_ratio",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "hl_ratio",
]
print(f"  Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")

# Drop NaN rows
df = df.dropna(subset=FEATURE_COLS + ["target"])
print(f"  Clean rows: {len(df):,}")

# ─── 3. Train / Test Split ─────────────────────────────────────────────────────
cutoff = len(df) - 24*90  # last 90 days = test
df_train = df.iloc[:cutoff]
df_test  = df.iloc[cutoff:]

X_train = df_train[FEATURE_COLS].values.astype(np.float32)
y_train = df_train["target"].values.astype(np.float32)
X_test  = df_test[FEATURE_COLS].values.astype(np.float32)
y_test  = df_test["target"].values.astype(np.float32)

print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# Clip extreme returns (>50% in 24h = data issue)
y_train = np.clip(y_train, -0.5, 0.5)
y_test  = np.clip(y_test,  -0.5, 0.5)

# Feature scaling
from sklearn.preprocessing import RobustScaler
scaler   = RobustScaler()
X_train  = scaler.fit_transform(X_train)
X_test   = scaler.transform(X_test)

# Clip extreme scaled values
X_train = np.clip(X_train, -5, 5)
X_test  = np.clip(X_test,  -5, 5)

with open(SCALER_PATH, "wb") as f:
    pickle.dump({"scaler": scaler, "feature_cols": FEATURE_COLS}, f)
print(f"  Scaler saved → {SCALER_PATH}")

# ─── 4. Build & Train MLP ──────────────────────────────────────────────────────
print("\nBuilding MLP (Dense 64 → 32 → 1)...")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(len(FEATURE_COLS),), name="input"),
    tf.keras.layers.Dense(64, activation="relu", name="dense1"),
    tf.keras.layers.Dense(32, activation="relu", name="dense2"),
    tf.keras.layers.Dense(1,  activation="linear", name="output"),
], name="btc_mlp")

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="mse",
    metrics=["mae"],
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)
lr_sched = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
)

print("Training...")
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=150,
    batch_size=512,
    callbacks=[early_stop, lr_sched],
    verbose=1,
)

# ─── 5. Evaluate ──────────────────────────────────────────────────────────────
print("\nEvaluating on test set...")
y_pred    = model.predict(X_test, verbose=0).flatten()
prices    = df_test["close"].values
curr_pr   = prices[:-1]
pred_pr   = curr_pr * (1 + y_pred[:-1])
actual_pr = prices[1:24*90:24][:len(pred_pr)]  # daily actual

mae_ret   = np.mean(np.abs(y_pred - y_test))
dir_corr  = np.mean(np.sign(y_pred) == np.sign(y_test))

print(f"  MAE (return): {mae_ret:.5f}  ({mae_ret*100:.3f}%)")
print(f"  Direction accuracy: {dir_corr*100:.1f}%")

# ─── 6. Export to ONNX ────────────────────────────────────────────────────────
print("\nExporting to ONNX...")
import tf2onnx, onnx

# Save as TF SavedModel via model.export() (Keras 3 style)
sm_path = str(OUT_DIR / "mlp_savedmodel")
model.export(sm_path)          # creates a TF2 SavedModel directory
print(f"  SavedModel exported → {sm_path}")

# Convert from SavedModel using CLI (most reliable for Keras 3)
import subprocess
result = subprocess.run([
    "python3", "-m", "tf2onnx.convert",
    "--saved-model", sm_path,
    "--output", str(ONNX_PATH),
    "--opset", "13",
    "--inputs", f"input:0[1,{len(FEATURE_COLS)}]",
], capture_output=True, text=True)
print(result.stdout[-500:] if result.stdout else "")
if result.returncode != 0:
    print("STDERR:", result.stderr[-500:])
    sys.exit(f"tf2onnx failed: {result.returncode}")
print(f"  ONNX saved → {ONNX_PATH}")

# Verify
onnx_model = onnx.load(str(ONNX_PATH))
onnx.checker.check_model(onnx_model)
print("  ✓ ONNX model valid")

# ─── 7. EZKL Proof Generation ────────────────────────────────────────────────
print("\n" + "="*60)
print("EZKL ZKML PROOF GENERATION")
print("="*60)

import ezkl

# Use a single representative input sample (last known BTC features)
sample_idx = -1  # last sample in test set (most recent)
x_sample   = X_test[sample_idx:sample_idx+1].astype(np.float32)  # shape (1, 18)

# EZKL expects input as {"input_data": [[...]]}
input_data = {"input_data": [x_sample.flatten().tolist()]}
with open(INPUT_PATH, "w") as f:
    json.dump(input_data, f)
print(f"  Sample input saved → {INPUT_PATH}")
print(f"  Input shape: {x_sample.shape}, values[:5]: {x_sample.flatten()[:5].tolist()}")

# Calibration data: 20 random samples from test set
np.random.seed(42)
cal_idx   = np.random.choice(len(X_test), size=20, replace=False)
cal_data  = {"input_data": X_test[cal_idx].flatten().tolist()}
# EZKL calibrate_settings expects list of inputs
cal_data_proper = {"input_data": [X_test[i].flatten().tolist() for i in cal_idx]}
with open(CAL_DATA_PATH, "w") as f:
    json.dump(cal_data_proper, f)
print(f"  Calibration data saved (20 samples) → {CAL_DATA_PATH}")

# Step 1: Generate settings
print("\n[EZKL 1/6] gen_settings...")
run_args = ezkl.PyRunArgs()
run_args.input_visibility  = "public"
run_args.output_visibility = "public"
run_args.param_visibility  = "private"   # weights private (ZK!)

res = ezkl.gen_settings(
    str(ONNX_PATH),
    str(SETTINGS_PATH),
    py_run_args=run_args,
)
print(f"  ✓ settings.json created  (result={res})")

# Step 2: Calibrate settings
print("[EZKL 2/6] calibrate_settings (this may take 1-2 min)...")
res = ezkl.calibrate_settings(
    str(INPUT_PATH),
    str(ONNX_PATH),
    str(SETTINGS_PATH),
    "resources",   # target: optimise for small proof size
)
print(f"  ✓ settings.json calibrated  (result={res})")

# Show settings
with open(SETTINGS_PATH) as f:
    settings = json.load(f)
logrows = settings.get("run_args", {}).get("logrows", "?")
print(f"  logrows = {logrows}  (circuit size = 2^{logrows})")

# Step 3: Compile circuit
print("[EZKL 3/6] compile_circuit...")
res = ezkl.compile_circuit(
    str(ONNX_PATH),
    str(COMPILED_PATH),
    str(SETTINGS_PATH),
)
print(f"  ✓ circuit compiled → {COMPILED_PATH}  (result={res})")

# Step 4: Get SRS (KZG trusted setup)
print("[EZKL 4/6] get_srs (downloading KZG params, may take ~30s)...")
res = ezkl.get_srs(str(SETTINGS_PATH), srs_path=str(SRS_PATH))
print(f"  ✓ kzg.srs saved → {SRS_PATH}  (result={res})")

# Step 5: Setup (generates pk.key and vk.key)
print("[EZKL 5/6] setup (proving key + verification key)...")
res = ezkl.setup(
    str(COMPILED_PATH),
    str(VK_PATH),
    str(PK_PATH),
    srs_path=str(SRS_PATH),
)
print(f"  ✓ pk.key → {PK_PATH}")
print(f"  ✓ vk.key → {VK_PATH}  (result={res})")

# Step 6: Generate witness + proof
print("[EZKL 6/6] gen_witness + prove...")
res_w = ezkl.gen_witness(
    str(INPUT_PATH),
    str(COMPILED_PATH),
    str(WITNESS_PATH),
)
print(f"  ✓ witness generated  (result={res_w})")

res_p = ezkl.prove(
    str(WITNESS_PATH),
    str(COMPILED_PATH),
    str(PK_PATH),
    str(PROOF_PATH),
    proof_type="single",
    srs_path=str(SRS_PATH),
)
print(f"  ✓ proof.json → {PROOF_PATH}  (result={res_p})")

# Verify the proof
print("\n[Verify] Checking proof...")
res_v = ezkl.verify(
    str(PROOF_PATH),
    str(SETTINGS_PATH),
    str(VK_PATH),
    srs_path=str(SRS_PATH),
)
print(f"  ✓ Proof verified: {res_v}")

# ─── 8. Summary ───────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("EZKL ARTIFACTS READY")
print("="*60)
for path, label in [
    (VK_PATH,       "vk.key          (Verification Key)"),
    (SETTINGS_PATH, "settings.json   (Circuit Settings)"),
    (PROOF_PATH,    "proof.json       (Sample ZK Proof)"),
    (SRS_PATH,      "kzg.srs          (KZG Reference String)"),
]:
    size = Path(path).stat().st_size if Path(path).exists() else 0
    ok   = "✓" if Path(path).exists() else "✗"
    print(f"  {ok}  {label:40s}  {size:,} bytes")

print(f"\nAll files in: {OUT_DIR}")
print("\nUpload these 4 files to Rainface → Share Model page:")
print("  vk.key, settings.json, proof.json, kzg.srs")
