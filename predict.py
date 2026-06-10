#!/usr/bin/env python3
"""
BTC 24-hour direction prediction using MLP v2 binary classifier.
Output: probability that BTC price is higher in 24 hours.

Usage:
    python3 predict.py
"""
import pickle, sys
from pathlib import Path

import numpy as np
import pandas as pd
import boto3, io
import onnxruntime as rt

S3_BUCKET   = "bitcoin-prediction-option4-production-654654488711"
S3_KEY      = "training-data/btc_hourly.parquet"
MODEL_PATH  = Path(__file__).parent / "mlp.onnx"
SCALER_PATH = Path(__file__).parent / "scaler.pkl"


def build_features(df_raw):
    df = df_raw.copy()
    df.columns = [c.lower() for c in df.columns]
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()

    df["close"]  = df["price"].astype(float) if "price" in df.columns else df["close"].astype(float)
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
    df["macd_norm"]   = (ema12 - ema26) / (df["close"] + 1e-9)
    df["macd_signal"] = df["macd_norm"].ewm(span=9).mean()

    df["vol_ratio"] = df["volume"] / (df["volume"].rolling(24).mean() + 1e-9) - 1
    df["hour_sin"]  = np.sin(2*np.pi * df.index.hour / 24)
    df["hour_cos"]  = np.cos(2*np.pi * df.index.hour / 24)
    df["dow_sin"]   = np.sin(2*np.pi * df.index.dayofweek / 7)
    df["dow_cos"]   = np.cos(2*np.pi * df.index.dayofweek / 7)
    df["hl_ratio"]  = (df["high"].astype(float) - df["low"].astype(float)) / (df["close"] + 1e-9)
    for lag in [1, 2, 3]:
        df[f"ret_lag{lag}"] = df["close"].pct_change(lag)
    return df


def main():
    print("BTC MLP v2 — 24-hour Direction Prediction")
    print("=" * 45)

    with open(SCALER_PATH, "rb") as f:
        d = pickle.load(f)
    scaler, FEATURE_COLS = d["scaler"], d["feature_cols"]
    sess = rt.InferenceSession(str(MODEL_PATH))

    print("Fetching latest BTC data from S3...")
    s3     = boto3.client("s3")
    obj    = s3.get_object(Bucket=S3_BUCKET, Key=S3_KEY)
    df_raw = pd.read_parquet(io.BytesIO(obj["Body"].read()))

    df = build_features(df_raw)
    df = df.dropna(subset=FEATURE_COLS)

    latest        = df.iloc[-1]
    current_price = float(latest["close"])
    ts            = df.index[-1]

    row    = df[FEATURE_COLS].iloc[-1:].values.astype(np.float32)
    row_sc = np.clip(scaler.transform(row), -5, 5).astype(np.float32)

    prob_up   = float(sess.run(None, {"input": row_sc})[0][0][0])
    prob_down = 1.0 - prob_up
    call      = "UP ↑" if prob_up >= 0.5 else "DOWN ↓"

    # Expected return = weighted average of conditional means from training data
    MEAN_UP_RET   =  0.01742   # avg 24h return on UP days (training set)
    MEAN_DOWN_RET = -0.01740   # avg 24h return on DOWN days (training set)
    expected_ret  = prob_up * MEAN_UP_RET + prob_down * MEAN_DOWN_RET
    pred_price    = current_price * (1 + expected_ret)

    print(f"\n  Timestamp      : {ts}")
    print(f"  Current BTC    : ${current_price:,.2f}")
    print(f"  Predicted 24h  : ${pred_price:,.2f}  ({expected_ret*100:+.2f}%)")
    print(f"  P(UP)          : {prob_up*100:.1f}%")
    print(f"  P(DOWN)        : {prob_down*100:.1f}%")
    print(f"  Call           : {call}")
    print()
    print("Key signals:")
    print(f"  RSI (14)   : {latest['rsi14']*100:.0f}  (oversold <30, overbought >70)")
    print(f"  24h return : {latest['ret_24h']*100:+.2f}%")
    print(f"  7d return  : {latest['ret_7d']*100:+.2f}%")
    print(f"  vs SMA-30d : {latest['vs_sma30']*100:+.2f}%")
    print(f"  BB pos     : {latest['bb_pct']:+.3f}")
    print()
    print("Model: Dense(64,ReLU)+Dropout → Dense(32,ReLU)+Dropout → Dense(1,Sigmoid)")
    print("Test accuracy: 51.9%  |  Baseline (always UP): 48.7%")


if __name__ == "__main__":
    main()
