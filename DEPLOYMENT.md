# Free-tier deployment options

The project has a small MobileNetV3-Small checkpoint (6,234,821 bytes; SHA-256 `5acce83e74552321e01a5dceb962f69441919a18885006cbb77cbdc9e900b93f`). The backend loads one CPU model, uses one Uvicorn worker, runs batch-one inference under `torch.inference_mode()`, and limits Torch/OpenMP/OpenBLAS/OpenCV threads to one. A local Windows process measurement recorded 316.3 MiB peak working set after importing FastAPI/PyTorch, loading the checkpoint, estimating severity, and running inference. Render Free has 512 MiB RAM and only 0.1 CPU, so this leaves about 196 MiB measured headroom locally. Linux and hosting overhead differ; this is a plausible fit, not a verified production guarantee. The 0.1 CPU can make cold starts and predictions slow.

Render Free is now selected in `render.yaml` as requested. Its instance sleeps after 15 minutes without inbound traffic, can take about a minute to wake, and has an ephemeral filesystem. SQLite history is therefore disposable on restart/sleep/redeploy. Use Supabase for history that must persist. Render's free-plan limits and memory allocation are documented at [Render Free](https://render.com/docs/free) and [Render Compute Plans](https://render.com/docs/compute-plans).

For a more dependable free-quota deployment with additional memory, this repository also includes a Dockerfile for Google Cloud Run. Configure 1 vCPU, 1 GiB RAM, concurrency 1 and maximum instances 1. Cloud Run has an always-free request-based allowance (currently 180,000 vCPU-seconds, 360,000 GiB-seconds and 2 million requests/month), but [Google requires a billing account](https://docs.cloud.google.com/free/docs/free-cloud-features?hl=en) and charges usage exceeding the allowance; set a budget alert and confirm the current pricing first. This is free only while usage stays within quota. See [Cloud Run pricing](https://cloud.google.com/run/pricing) and [service memory settings](https://cloud.google.com/run/docs/configuring/services/memory-limits).

## 1. Prepare GitHub without datasets or model weights

From PowerShell in the project root:

```powershell
git status --short
git add .
git status --short
```

Check that your dataset folders, `.venv`, `paddyguard.sqlite3`, and `artifacts/paddyguard.pt` are not staged. `.gitignore` and `.dockerignore` exclude these large/local artifacts. If this is a new Git repository, create an empty GitHub repository and push:

```powershell
git init -b main
git add .
git commit -m "Prepare PaddyGuard-TN deployment"
git remote add origin https://github.com/YOUR-GITHUB-NAME/paddyguard-tn.git
git push -u origin main
```

Skip `git init` if this folder already has a repository. GitHub authentication is required to push.

## 2. Host the trained model artifact

The model is not committed to source control. Create a GitHub release with tag `paddyguard-model-v1` and attach `artifacts/paddyguard.pt`. For a public release, its direct download URL follows this pattern:

```text
https://github.com/YOUR-GITHUB-NAME/paddyguard-tn/releases/download/paddyguard-model-v1/paddyguard.pt
```

The SHA-256 for the current checkpoint is:

```text
5acce83e74552321e01a5dceb962f69441919a18885006cbb77cbdc9e900b93f
```

The backend downloads only over HTTPS and verifies the full SHA-256 before loading. After replacing/retraining the checkpoint, calculate the new digest:

```powershell
(Get-FileHash .\artifacts\paddyguard.pt -Algorithm SHA256).Hash.ToLower()
```

Update the `MODEL_SHA256` setting on the host. For private storage, set `MODEL_URL` and `MODEL_DOWNLOAD_TOKEN` as backend-only secrets; never expose the token in a `VITE_` variable.

## 3. Deploy the backend on Render Free (best-effort)

1. In Render, create a Blueprint using the GitHub repository. Review `render.yaml`; it selects the `free` plan, Python 3.11, one Uvicorn worker and `/readyz`.
2. Set `MODEL_URL` to the release download URL. Keep the configured SHA-256, and set `CORS_ORIGINS` to the exact Vercel site origin (for example `https://paddyguard-tn.vercel.app`, no trailing slash). Set `MODEL_DOWNLOAD_TOKEN` only for private model hosting.
3. If prediction history must survive Render restarts, configure Supabase by setting `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` in Render. Otherwise the default SQLite database is under `/tmp` and history is temporary.
4. Wait for deployment. Check `https://YOUR-RENDER-SERVICE.onrender.com/healthz` and `/readyz`. `/readyz` must return 200 with `model: ready`; `/api/health` should report `model_ready: true`.

The API never substitutes a placeholder prediction when the model cannot load. It returns unavailable/not-ready instead. If Render Free repeatedly fails readiness, restarts due to memory, or is too slow, use Cloud Run below rather than lowering confidence rules or faking results.

## 4. Deploy the backend on Google Cloud Run free quota (recommended for more memory)

This option requires a Google account, a Google Cloud project, an enabled billing account, and the `gcloud` CLI. Usage is free only inside the monthly allowance; configure a billing budget alert. Choose the `asia-south1` Mumbai region when available.

1. Push the source to GitHub and publish the model release as above. Install and authenticate the Google Cloud CLI, select your project, and enable billing/Cloud Run APIs if prompted.
2. From the project root, deploy the Dockerfile (datasets and weights are excluded from its build context):

```powershell
gcloud run deploy paddyguard-api `
  --source . `
  --project YOUR-GCP-PROJECT-ID `
  --region asia-south1 `
  --allow-unauthenticated `
  --memory 1Gi `
  --cpu 1 `
  --concurrency 1 `
  --max 1 `
  --min 0 `
  --timeout 300 `
  --set-env-vars "APP_ENV=production,MODEL_PATH=/tmp/paddyguard.pt,MODEL_URL=https://github.com/YOUR-GITHUB-NAME/paddyguard-tn/releases/download/paddyguard-model-v1/paddyguard.pt,MODEL_SHA256=5acce83e74552321e01a5dceb962f69441919a18885006cbb77cbdc9e900b93f,DATABASE_PATH=/tmp/paddyguard.sqlite3,TORCH_NUM_THREADS=1,OMP_NUM_THREADS=1,MKL_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,OPENCV_FOR_THREADS_NUM=1,ABSTAIN_THRESHOLD=0.55,CORS_ORIGINS=https://YOUR-VERCEL-SITE.vercel.app"
```

Replace project, GitHub owner/repository, and Vercel origin values. For private model hosting, configure `MODEL_URL` and `MODEL_DOWNLOAD_TOKEN` through Secret Manager/Cloud Run secrets instead of command-line environment arguments. The instance uses one request at a time to bound inference memory and scales to zero. Cloud Run's local SQLite is also ephemeral; set Supabase secrets if history persistence is required.

After deploy, copy the service URL printed by `gcloud` and verify `/healthz`, `/readyz`, and `/api/health` until model readiness is true. The Cloud Run API is not publicly verified until these checks pass.

## 5. Deploy the React frontend on Vercel

1. Import the GitHub repository into Vercel, set **Root Directory** to `frontend`, framework **Vite**, Node `22.x`, build command `npm run build`, output `dist`.
2. Set `VITE_API_BASE_URL` to the deployed backend URL plus `/api`, for example `https://paddyguard-api-abc.a.run.app/api`. This is compiled into the frontend, so redeploy after changing it.
3. Set the backend's `CORS_ORIGINS` to the Vercel production origin, then redeploy the backend if you changed it.
4. Upload a clear labeled-field paddy leaf image and check the browser Network panel for a successful `POST /api/predict`. Verify the disease class, confidence, severity percentage/label and advisory source links. A low-confidence response must abstain.

Vercel frontend configuration is in `frontend/vercel.json`. No deployed image upload has been claimed or verified from this local workspace; that final end-to-end check requires the user's hosting accounts and public service URLs. Keep in mind Render Free can sleep and loses SQLite data; Cloud Run can also scale to zero and its local filesystem is disposable.
