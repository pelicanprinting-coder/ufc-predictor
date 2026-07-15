# Deploy to Streamlit Community Cloud

Free hosting from Streamlit that natively supports this app's WebSocket protocol.

## One-time setup (~5 minutes)

1. Go to https://share.streamlit.io/signup and sign in **with GitHub** (use the same GitHub account this repo is forked to: `pelicanprinting-coder`).
2. Click **New app** → **Deploy a public app from GitHub**.
3. Fill in:
   - **Repository:** `pelicanprinting-coder/ufc-predictor`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL (subdomain):** anything you like (e.g. `cagepicks-ufc`)
4. Click **Advanced settings** → **Secrets** and paste:
   ```toml
   ODDS_API_KEY = "your-odds-api-key-here"
   TOKEN_GITHUB = ""
   ```
   Get a free Odds API key at https://the-odds-api.com (500 requests/month free tier). You can also leave `ODDS_API_KEY` empty — the app runs without live odds.
5. Click **Deploy**.

First build takes ~4 minutes (installs xgboost, lightgbm, catboost). Subsequent visits are fast.

## Updating the app

Push commits to `main` — Streamlit Cloud auto-redeploys.

## Notes

- The `.streamlit/config.toml` in this repo sets a dark theme and disables telemetry.
- The `dist/` folder is only there from an earlier pplx.app publish attempt and is not used by Streamlit — safe to ignore or delete.
- If deployment errors out on package resolution, check the `requirements.txt` versions haven't been superseded by ML library updates.
