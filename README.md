# PaddyGuard-TN

Cross-dataset paddy leaf diagnosis with confidence-threshold abstention, visual severity estimation and bilingual advice.

## Data inventory and honest model status

The bundled `paddy-disease-classification` dataset contains 10,407 labeled images across ten classes in `train.csv` and `train_images/`. The folder labels match the CSV. `test_images/` has 3,469 images, but `sample_submission.csv` has no labels, so those images are excluded from evaluation. The trainer removes byte-identical duplicates before making deterministic stratified 70/15/15 train/validation/test splits from the labeled set.

The ten actual image classes and counts are bacterial_leaf_blight (479), bacterial_leaf_streak (380), bacterial_panicle_blight (337), blast (1,738), brown_spot (965), dead_heart (1,442), downy_mildew (620), hispa (1,594), normal (1,764), and tungro (1,088). Some labels describe pest damage or symptoms rather than diseases; the classifier reports the dataset's label and the advisory record flags provisional mappings. Train using `python -m backend.train --data paddy-disease-classification --epochs 5 --batch-size 24 --cpu-threads 4`. The script detects `train_images/`, uses MobileNetV3-Small with pretrained weights, class-balanced sampling, augmentation, validation-based checkpoint selection, and held-out metrics. It writes `artifacts/paddyguard.pt` and `artifacts/metrics.json`.

The currently saved checkpoint is the best validation checkpoint from a training run that stopped during epoch 4; it is not represented as a completed five-epoch run. Its recorded best validation accuracy was 75.81%. The deterministic held-out test evaluation for that checkpoint is 72.77% accuracy, 0.7228 macro precision, 0.7413 macro recall, and 0.7274 macro F1 on 1,550 images. Recreate that evaluation without retraining via `python backend/evaluate_checkpoint.py --data paddy-disease-classification --checkpoint artifacts/paddyguard.pt`. The unlabeled competition `test_images/` were not included in evaluation. Resume fine-tuning from the saved checkpoint with `python -m backend.train --data paddy-disease-classification --epochs 2 --batch-size 24 --cpu-threads 4 --resume`; `--epochs` means additional epochs when `--resume` is set.

## Windows / VS Code setup

1. Install Python 3.11 and Node.js 22.12 or newer; open this folder in VS Code.
2. In a terminal run `py -3.11 -m venv .venv`, then `.venv\Scripts\Activate.ps1`.
3. Install API dependencies: `python -m pip install -r requirements.txt`.
4. Install frontend dependencies: `cd frontend; npm install; cd ..`.
5. Copy `.env.example` to `.env`; by default the app uses local SQLite.
6. Start API: `python -m uvicorn backend.main:app --reload`.
7. In a second terminal run `cd frontend; npm run dev` and open the displayed URL.
8. Train with the existing dataset using `python -m backend.train --data paddy-disease-classification --epochs 5`, then restart the API. The model and metrics are saved under `artifacts/`.

The API also serves the built frontend when `frontend/dist` exists. For production, build with `cd frontend; npm run build` and run `python -m uvicorn backend.main:app`.

## Public deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for free-tier deployment steps, model release/checksum setup, CORS, production environment variables, and post-deployment upload verification. `render.yaml` uses Render Free. A local CPU inference smoke test measured 316 MiB peak process working set on Windows against Render's 512 MiB limit; the remaining headroom and Render's 0.1 CPU are limited, so the deployment guide also gives a 1 GiB Cloud Run option that stays within its published monthly free allowance if usage stays below the limits (billing account required). Neither host has been deployed from this workspace.

## Features and limits

- English/Tamil dashboard, image upload, abstention, severity percentage and history.
- Severity is a color-based lesion estimate and is **not** a validated disease segmentation model; lighting and leaf background affect it.
- Bilingual official-source summaries for blast, brown spot, bacterial leaf blight, bacterial leaf streak, and tungro link to TNAU pages in `official_advisories.json`. No pesticide dose is generated. Other classes use the supplied CSV only as a **synthetic starter fallback**, with that status shown in the UI and its stored references preserved.
- SQLite is the default. Supabase is selected when both URL and service key are configured. Create the table first with columns matching this schema (use `advisory` as JSONB):

```sql
create table paddyguard_predictions (
  id text primary key, created_at timestamptz not null, filename text not null,
  class_name text, confidence double precision not null, abstained boolean not null,
  affected_pct double precision not null, severity text not null,
  language text not null, advisory jsonb
);
```

- Use a field expert to confirm diagnosis and consult current local label directions before treatment.

## API

- `GET /api/health`, `GET /api/classes`, `GET /api/stats`, `GET /api/history`
- `POST /api/predict` multipart field `file`; optional `language=en|ta`
- `DELETE /api/history`

The classifier was trained from the bundled labeled images. The reported metrics come from the deterministic stratified held-out split; the unlabeled competition images were not used for evaluation. The deployment test must still be performed after you connect your public Vercel and Render URLs.
