# BTC MLP Proxy Model — ZKML Verified

A small Neural Network (Dense 64→32→1, **3,329 params**) trained on Bitcoin hourly OHLCV data to predict the 24-hour price return. Built specifically for [zero-knowledge machine learning (ZKML)](https://ezkl.xyz) verification via EZKL.

## Model

| Attribute | Value |
|-----------|-------|
| Architecture | Dense(64, ReLU) → Dense(32, ReLU) → Dense(1, Linear) |
| Parameters | 3,329 |
| Input | 18 BTC technical features |
| Output | 24-hour % return |
| Training data | 14,696 hourly BTC candles (Jun 2024 – Jun 2026) |
| Test MAE | 1.45% (return) |
| EZKL circuit | logrows = 15 (2^15 = 32,768 rows) |

## Features

```
ret_1h, ret_4h, ret_24h, ret_7d       — price returns
vol_24h, vol_7d                        — rolling volatility
bb_upper, bb_pct                       — Bollinger Bands
vs_sma7, vs_sma30                      — SMA ratio signals
rsi14, macd_norm                       — momentum
vol_ratio                              — volume anomaly
hour_sin, hour_cos, dow_sin, dow_cos   — time cyclicals
hl_ratio                               — intraday range
```

## ZKML Artifacts (EZKL v23.0.5)

| File | Description |
|------|-------------|
| `vk.key` | Verification Key |
| `settings.json` | EZKL circuit settings |
| `proof.json` | Sample ZK proof (verified) |
| `kzg.srs` | KZG Structured Reference String |

Proof verified: `ezkl.verify()` → `True`

## Usage

```bash
pip install onnxruntime boto3 pandas numpy scikit-learn pyarrow

# Run prediction
python3 predict.py

# Retrain + regenerate EZKL proofs
python3 train_and_prove.py
```

## Competition

Built for the [Rainface.finance](https://rainface.finance) Bitcoin prediction competition.
