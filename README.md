# BTC MLP v2 — Direction Classifier + ZKML Proof

A small Neural Network trained on Bitcoin hourly OHLCV data to predict the **direction** (UP/DOWN) of BTC price over the next 24 hours. Built for [zero-knowledge machine learning (ZKML)](https://ezkl.xyz) verification via EZKL.

## Model

| Attribute | Value |
|-----------|-------|
| Architecture | Dense(64,ReLU)+Dropout(0.3) → Dense(32,ReLU)+Dropout(0.3) → Dense(1,Sigmoid) |
| Parameters | 3,585 |
| Input | 22 BTC technical features |
| Output | Probability that price is higher in 24 hours (0 = DOWN, 1 = UP) |
| Training data | 14,696 hourly BTC candles (Jun 2024 – Jun 2026) |
| Test accuracy | 51.9% direction (vs 48.7% baseline) |
| EZKL circuit | logrows = 15 (2^15 rows) |

## Output Interpretation

```
P(UP) > 0.55  →  Confident UP
P(UP) = 0.45–0.55  →  Neutral / uncertain
P(UP) < 0.45  →  Confident DOWN
```

## Features (22)

```
ret_1h, ret_4h, ret_24h, ret_7d        — price returns
vol_24h, vol_7d                         — rolling volatility  
bb_upper, bb_pct                        — Bollinger Bands
vs_sma7, vs_sma30                       — SMA ratio signals
rsi14, macd_norm, macd_signal           — momentum
vol_ratio                               — volume anomaly
hour_sin, hour_cos, dow_sin, dow_cos    — time cyclicals
hl_ratio                                — intraday range
ret_lag1, ret_lag2, ret_lag3            — short-lag returns
```

## ZKML Artifacts (EZKL v23.0.5)

| File | Description |
|------|-------------|
| `vk.key` | Verification Key |
| `settings.json` | EZKL circuit settings |
| `proof.json` | Sample ZK proof (verified ✓) |
| `kzg.srs` | KZG Structured Reference String |

## Usage

```bash
pip install onnxruntime boto3 pandas numpy scikit-learn pyarrow

# Run live prediction
python3 predict.py
```

## Competition

Built for the [Rainface.finance](https://rainface.finance) Bitcoin prediction competition.
