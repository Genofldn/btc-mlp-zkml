#!/usr/bin/env python3
"""
BTC 24-hour price prediction using the MLP proxy model.
Fetches latest BTC data from S3, engineers features, and outputs prediction.

Usage:
    python3 predict.py
    python3 predict.py --current-price 61658
"""

import argparse, pickle, sys
from pathlib import Path

import numpy as np
import pandas as pd
import boto3, io
import onnxruntime as rt

S3_BUCKET  = "bitcoin-prediction-option4-production-654654488711"
S3_KEY     = "training-data/btc_hourly.parquet"
MODEL_PATH = Path(__file__).parent / "mlp_fixed.onnx"
SCALER_PATH= Path(__file__).parent / "scaler.pkl"


def build_features(df_raw):
    df = df_raw.copy()
    df.columns = [c.lower() for c in df.columns]
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()
    price_col = "price" if "price" in df.columns else "close"
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

    delta = df["close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi14"] = (100 - 100 / (1 + gain / (loss + 1e-9))) / 100.0

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

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-price", type=float, help="Override current BTC price")
    args = parser.parse_args()

    print("BTC MLP Proxy — 24-hour Price Prediction")
    print("=" * 45)

    # Load model + scaler
    with open(SCALER_PATH, "rb") as f:
        d = pickle.load(f)
    scaler, FEATURE_COLS = d["scaler"], d["feature_cols"]
    sess = rt.InferenceSession(str(MODEL_PATH))

    # Fetch data
    print("Fetching latest BTC data from S3...")
    s3  = boto3.client("s3")
    obj = s3.get_object(Bucket=S3_BUCKET, Key=S3_KEY)
    df_raw = pd.read_parquet(io.BytesIO(obj["Body"].read()))
    print(f"  {len(df_raw):,} hourly candles  [{df_raw.index[0]} → {df_raw.index[-1]}]")

    df = build_features(df_raw)
    df = df.dropna(subset=FEATURE_COLS)

    latest = df.iloc[-1]
    current_price = args.current_price or float(latest["close"])
    ts = df.index[-1]

    row = df[FEATURE_COLS].iloc[-1:].values.astype(np.float32)
    row_scaled = np.clip(scaler.transform(row), -5, 5).astype(np.float32)

    pred_return = sess.run(None, {"input": row_scaled})[0][0][0]
    pred_price  = current_price * (1 + pred_return)
    direction   = "UP ↑" if pred_return > 0 else "DOWN ↓"
    change_pct  = pred_return * 100

    print(f"\n  Timestamp  : {ts}")
    print(f"  Current BTC: ${current_price:,.2f}")
    print(f"  Pred return: {change_pct:+.3f}%")
    print(f"  Pred 24h   : ${pred_price:,.2f}")
    print(f"  Direction  : {direction}")
    print()

    # Key features driving the prediction
    print("Key signals:")
    print(f"  RSI (14)      : {latest['rsi14']*100:.1f}")
    print(f"  24h return    : {latest['ret_24h']*100:+.2f}%")
    print(f"  7d return     : {latest['ret_7d']*100:+.2f}%")
    print(f"  vs SMA-7d     : {latest['vs_sma7']*100:+.2f}%")
    print(f"  vs SMA-30d    : {latest['vs_sma30']*100:+.2f}%")
    print(f"  BB position   : {latest['bb_pct']:+.3f}")


if __name__ == "__main__":
    main()
