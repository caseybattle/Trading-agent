# Railway Deployment — Setup Guide

## Step 1 — Create Railway account
Go to https://railway.app and sign up with GitHub.

## Step 2 — Create project from GitHub repo
- New Project → Deploy from GitHub repo
- Select the `prediction-market-bot` repository
- Railway auto-detects `railway.json` and configures the **bot-scanner** cron service (runs every 30 min)

## Step 3 — Add environment variables
In Railway dashboard → your service → Variables, add:

| Variable | Value |
|---|---|
| `KALSHI_API_KEY_ID` | your Kalshi API key ID |
| `KALSHI_USE_DEMO` | `false` |
| `KALSHI_PRIVATE_KEY_CONTENTS` | (see Step 4) |
| `BANKROLL` | `10` |

## Step 4 — Upload your private key
Railway has no persistent filesystem, so paste the PEM contents as an env var.

```bash
# On your local machine, print the key with literal \n so it fits in one line:
python -c "print(open('kalshi_private_key.pem').read().replace('\n', '\\n'))"
```

Copy the output and paste it as the value of `KALSHI_PRIVATE_KEY_CONTENTS`.
The bot writes it to a temp file at startup automatically.

## Step 5 — Deploy
Click Deploy (or push to the connected branch). The bot-scanner cron starts automatically.

Check logs in Railway dashboard → your service → Deployments.

---

## Adding watchdog and live-optimizer services

`railway.json` configures one service (the bot-scanner). To add the other two:

1. In your Railway project, click **+ New Service** → **GitHub repo** (same repo)
2. Set the **Start Command** for each:
   - Watchdog: `python scripts/watchdog.py` — set cron `0 8 * * *`
   - Optimizer: `python scripts/live_optimizer.py` — set cron `0 */6 * * *`
3. Copy the same env vars to each service (or use Railway's shared Variables)

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `[AUTH] Failed to load private key` | Check `KALSHI_PRIVATE_KEY_CONTENTS` — newlines must be `\n` not literal newlines |
| `ModuleNotFoundError` | Check `requirements.txt` covers the failing import |
| Service shows 0 deployments | Verify `railway.json` has a `startCommand` at the top-level `deploy` key (not nested under `services`) |
| No trades placed | Confirm `KALSHI_USE_DEMO=false` and `KALSHI_API_KEY_ID` is the production key |
