

---


```
PORT           = 10000        (Render sets this automatically)
ENV            = production
LOG_LEVEL      = INFO
AI_CACHE_TTL   = 900
CORS_ORIGINS   = *            (or your frontend URL)
```

Optional API keys (for more news sources):
```
ALPHA_VANTAGE_KEY = your_key
FINNHUB_KEY       = your_key
```

### Step 4 — Verify Deployment
```bash
curl https://your-app.onrender.com/health
# Expected: {"status":"ok","version":"4.0.0-prod","mode":"lightweight"}
```

---



---

## Local Testing (before deploy)

```bash
# Install deps
pip install -r requirements.txt

# Create .env
cp .env.example .env

# Run
uvicorn main:app --host 0.0.0.0 --port 10000

# Test
curl http://localhost:10000/health
curl http://localhost:10000/analyze -X POST \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL"}'
```

---

## API Endpoints

| Method | Path | Description | Response Time |
|--------|------|-------------|---------------|
| GET | `/health` | Service health | ~10ms |
| POST | `/analyze` | Quick rule-based signal | ~500ms |
| POST | `/predict` | Full ML prediction | ~5–15s* |
| GET | `/regime/{symbol}` | Market regime | ~2s |
| GET | `/sentiment/{symbol}` | News sentiment | ~2s |
| POST | `/portfolio` | Portfolio optimization | ~3s |
| GET | `/risk/{symbol}` | Risk assessment | ~1s |
| POST | `/backtest` | Strategy backtest | ~10s* |
| GET | `/indicators/{symbol}` | Technical indicators | ~2s |
| GET | `/signal/{symbol}` | Combined AI signal | ~2s |
| GET | `/macro` | Macro snapshot | ~3s |
| WS | `/ws` | Real-time stream | — |

*Cached after first call (15 min TTL)

---

## RAM Usage Estimate (Free Tier: 512MB)

| Component | RAM |
|-----------|-----|
| FastAPI + uvicorn | ~50MB |
| pandas + numpy | ~80MB |
| XGBoost + LightGBM | ~60MB |
| scikit-learn | ~30MB |
| matplotlib | ~40MB |
| Data cache (5 symbols) | ~20MB |
| **Total** | **~280MB** ✅ |

*(Old version with torch: ~1.8GB ❌)*

---

## Common Issues

**`ModuleNotFoundError: No module named 'lstm_model'`**
→ Make sure you deleted `lstm_model.py` OR the import is not in any kept file.
→ `ensemble_model.py` no longer imports it.

**`Address already in use`**
→ Render uses the `$PORT` env variable. The `main.py` reads it correctly.

**`Memory exceeded`**
→ Check you're not importing torch anywhere. Search: `grep -r "import torch" .`

**Cold start > 30s**
→ Normal on free tier (sleep after 15 min inactivity). First request warms up.
→ Use `/health` endpoint as a keep-alive ping (UptimeRobot free plan works).
