# Railway Deployment — 5-Step Setup

## Step 1 — Create Railway account
Go to https://railway.app and sign up with GitHub.

## Step 2 — Create project from GitHub repo
- New Project → Deploy from GitHub repo
- Select the `prediction-market-bot` repository
- Railway auto-detects `railway.json` and creates three cron services:
  - `bot-scanner` — runs every 30 min
  - `watchdog` — runs daily at 8 AM UTC
  - `live-optimizer` — runs every 6 hours

## Step 3 — Add environment variables
In Railway dashboard → your project → Variables, add:

| Variable | Value |
|---|---|
| `KALSHI_API_KEY_ID` | your Kalshi API key ID |
| `KALSHI_USE_DEMO` | `false` |
| `KALSHI_PRIVATE_KEY_CONTENTS` | (see Step 4) |

## Step 4 — Upload your private key
Railway has no persistent filesystem, so paste the PEM contents as an env var.

```bash
# On your local machine, print the key with literal \n so it fits in one line:
python -c "print(open('kalshi_private_key.pem').read().replace('\n', '\\n'))"
```

Copy the output and paste it as the value of `KALSHI_PRIVATE_KEY_CONTENTS`.
The bot writes it to a temp file at startup automatically.

**Alternative:** Use a Railway Volume (Persistent Storage) and set
`KALSHI_PRIVATE_KEY_PATH=/data/kalshi_private_key.pem`, then upload the file.

## Step 5 — Deploy
Click Deploy (or push to the connected branch). Railway starts all three cron jobs.

Check logs in Railway dashboard → each service → Deployments to confirm the
bot scanner is finding markets and the watchdog is passing health checks.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `[AUTH] Failed to load private key` | Check `KALSHI_PRIVATE_KEY_CONTENTS` — make sure newlines are `\n` not literal newlines |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` locally to verify; push updated requirements.txt |
| Bot exits immediately | Check `--once` flag in railway.json startCommand — correct, cron jobs always exit after one run |
| No trades placed | Set `KALSHI_USE_DEMO=false` and verify `KALSHI_API_KEY_ID` is the production key |
