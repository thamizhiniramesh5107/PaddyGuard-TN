# Public deployment: Vercel + Render

This repository is prepared for a Vercel static frontend and a Render FastAPI service. Publishing is not performed from this workspace because it requires your GitHub, Vercel, and Render account authorization and (for the recommended Render plan) billing approval. Deployment is **not live** until you complete the account steps below and verify the URLs.

## 1. Keep data and weights out of Git

The root `.gitignore` excludes the dataset directories, SQLite database, virtual environment, frontend build output, and `artifacts/*.pt`. The small `artifacts/metrics.json` evaluation report stays trackable. The trained checkpoint is about 6 MB and is downloaded by Render over HTTPS; its SHA-256 is verified before PyTorch loads it.

Open PowerShell in the project folder. If this folder is not already a Git repository, initialize it once:

```powershell
git init -b main
git status --short
git add .
git status --short
```

Confirm that neither `paddy-disease-classification/`, `artifacts/paddyguard.pt`, `.venv/`, nor `paddyguard.sqlite3` appears in the staged file list. Then create an **empty** repository in GitHub (private or public), and push:

```powershell
git commit -m "Prepare PaddyGuard-TN for deployment"
git remote add origin https://github.com/YOUR-GITHUB-NAME/paddyguard-tn.git
git push -u origin main
```

If the project is already a Git repository, skip `git init` and add the GitHub remote only if it does not already exist. GitHub login/authorization is required for the push.

## 2. Publish the model artifact for startup download

In GitHub, create a release with tag `paddyguard-model-v1` and attach the local file `artifacts/paddyguard.pt` as a release asset. The checkpoint is a model artifact, not source or a dataset; do not `git add -f` it. For a public release, set Render's `MODEL_URL` to:

```text
https://github.com/YOUR-GITHUB-NAME/paddyguard-tn/releases/download/paddyguard-model-v1/paddyguard.pt
```

The current checkpoint SHA-256 is:

```text
5acce83e74552321e01a5dceb962f69441919a18885006cbb77cbdc9e900b93f
```

`render.yaml` already carries this checksum. If you retrain or replace the checkpoint, calculate its new hash with:

```powershell
(Get-FileHash .\artifacts\paddyguard.pt -Algorithm SHA256).Hash.ToLower()
```

Update Render's `MODEL_SHA256` to that value before redeploying. For a private GitHub release, use the GitHub release asset API URL (`https://api.github.com/repos/OWNER/REPO/releases/assets/ASSET_ID`) for `MODEL_URL`, provide a read-only GitHub token in Render's secret `MODEL_DOWNLOAD_TOKEN`, and keep `MODEL_SHA256` set. Never put that token in Vercel or a `VITE_` variable. The API refuses non-HTTPS model URLs and hash mismatches; `/readyz` stays unavailable if the checkpoint cannot be downloaded or loaded.

## 3. Deploy the frontend to Vercel

1. Sign in to Vercel and import the GitHub repository. Grant Vercel access to that repository.
2. Set **Root Directory** to `frontend` and keep **Framework Preset** as Vite.
3. Use Node.js `22.x` (the project pins Vite-compatible tooling in `frontend/.nvmrc`), build command `npm run build`, and output directory `dist`.
4. Add the production environment variable `VITE_API_BASE_URL` with a temporary value `https://RENDER-SERVICE-NAME.onrender.com/api`. Replace it with the actual Render service URL after step 4, then redeploy. Vite embeds this value at build time; changing it requires a new deployment.
5. Deploy and note the production origin, for example `https://paddyguard-tn.vercel.app`. Do not use a preview URL as the only allowed origin if you also need production.

`frontend/vercel.json` provides SPA fallback routing. API requests go directly to the Render URL; the Vite development proxy remains `/api` → `localhost:8000` for local development.

## 4. Deploy the backend to Render

1. In Render, create a **Blueprint** from the same GitHub repository and review `render.yaml` before applying it.
2. The Blueprint uses Python 3.11, installs the CPU-only PyTorch wheels, starts one Uvicorn worker, and checks `/readyz`. One worker avoids duplicating PyTorch memory.
3. Enter the `MODEL_URL` from step 2 and set `CORS_ORIGINS` to the exact Vercel origin from step 3, without a trailing slash. Multiple allowed origins can be comma-separated (for example production and preview domains).
4. If using Supabase, set both `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` in Render only. Create the `paddyguard_predictions` table using the SQL in the project README. Never expose the service key in Vercel. If Supabase is not configured, the Blueprint uses SQLite under `/tmp`; that is suitable only for a demo because Render's ephemeral filesystem can be reset on restart/redeploy.
5. Review the compute plan and approve the cost in Render. The Blueprint selects Render's legacy `standard` plan (1 CPU, 2 GB RAM) because the PyTorch backend is not a reliable fit for the 512 MB free service. Render currently maps this plan to `1c-2g`; check current billing before you approve it.
6. Wait for the deploy and confirm Render logs show successful model initialization. Test `https://RENDER-SERVICE-NAME.onrender.com/healthz`, then `/readyz` and `/api/health`. `/readyz` must return HTTP 200 with `model: ready` before connecting the website.

The model is downloaded into `/tmp/paddyguard.pt` at startup and checksum-verified. The artifact is small, so restarts may download it again. You can change `MODEL_PATH` to a mounted persistent disk if you configure one, but a disk is not required for model startup.

## 5. Connect and verify the public website

1. In Vercel → Project → Settings → Environment Variables, set `VITE_API_BASE_URL` to the exact Render base URL plus `/api`, for example `https://paddyguard-api.onrender.com/api`.
2. Redeploy the Vercel project so the value is compiled into the Vite bundle.
3. Open the public Vercel domain, upload a clear paddy leaf image, and check for a dataset class label, confidence, approximate severity/affected-area percentage, and a disease-specific advisory with links when a verified TNAU source exists. If confidence is below the configured threshold, the app must abstain.
4. Confirm the browser's network panel shows a successful `POST` to the Render `/api/predict` endpoint and that Render's `/api/health` reports `model_ready: true`.

The workspace has passed a local model-backed upload smoke test, but no public image upload or live Vercel/Render deployment can be verified until you complete these account steps. The app's history is shared among visitors (there is no user sign-in or per-user isolation); use Supabase only if this shared-history behavior is acceptable. Production history deletion is protected by the Render-generated `HISTORY_ADMIN_TOKEN`; the dashboard asks for that token before clearing the shared history.
