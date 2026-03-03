# RawArchive

RawArchive is a Colab-first platform to turn Instagram chat exports into a personalized reply model.

Made by [@yashas.vm](https://github.com/YashasVM)

## What This Project Does

1. Upload one or more Instagram `.json` export files.
2. Parse and normalize message data.
3. Build a training bundle (`bun_...`) with train/validation samples.
4. Fine-tune a LoRA adapter on `Qwen/Qwen2.5-3B-Instruct` in Google Colab.
5. Download `adapter.zip`.
6. Register the adapter in the app (`mdl_...`).
7. Use local or Colab inference chat with the trained style.

## Project Structure

- `app/` FastAPI backend + frontend UI.
- `colab/` notebooks and training script.
- `scripts/` local inference scripts.
- `tests/` parser/builder/API tests.
- `data/` runtime artifacts (ignored in git).

## Prerequisites

- Python 3.11+
- Git
- Google Colab account
- (Optional) Cloudflare/ngrok tunnel for Colab access to local backend

## Step-By-Step: Run Locally

1. Create and activate virtual env:

```powershell
cd C:\Users\YashasVM\Downloads\code\LLM
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Start API + web app:

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

4. Open web app:

- `http://127.0.0.1:8000`

## Step-By-Step: Build a Training Bundle (`bun_...`)

1. In Step 1 UI, upload one or many Instagram `.json` files.
2. Wait for parse stats.
3. In Step 2, select target user and click **Build Bundle**.
4. Copy the generated `bundle_id` (`bun_...`).

## Step-By-Step: Train on Colab (Fastest)

Use notebook: `colab/train_lora_ultrafast.ipynb`

1. Upload/open the notebook in Colab.
2. Select **Runtime -> Change runtime type -> T4 GPU**.
3. Enter:
   - `BASE_URL` (your public tunnel URL)
   - `BUNDLE_ID` (your `bun_...`)
4. Run cells in order.
5. Download `adapter.zip` when training completes.

### If You Need a Tunnel

Run in local terminal (second tab):

```powershell
cloudflared tunnel --url http://127.0.0.1:8000
```

Use the provided `https://...trycloudflare.com` as `BASE_URL` in Colab.

## Step-By-Step: Register Trained Model (`mdl_...`)

1. Move downloaded `adapter.zip` to:

`C:/Users/YashasVM/Downloads/code/LLM/data/models/adapter.zip`

2. In web app Step 4, fill:
- Adapter URI: `local://C:/Users/YashasVM/Downloads/code/LLM/data/models/adapter.zip`
- Validation Loss: numeric only (example: `3.83`)
- Style Score: numeric only (example: `0.80`)

3. Click **Register Model**.
4. Save returned `model_id` (`mdl_...`).

## Step-By-Step: Chat With The Trained Adapter

### Option A: Local Chat (Windows)

1. Install inference deps:

```powershell
pip install -r requirements.inference.txt
```

2. Run:

```powershell
python scripts\chat_local.py --model-id mdl_bb8e7abb4c
```

3. Commands:
- `/reset` clear history
- `/exit` quit

### Option B: Colab Chat

Use notebook: `colab/chat_adapter_easy.ipynb`

1. Open in Colab.
2. Upload `adapter.zip`.
3. Run all cells.
4. Chat in last cell.

## API Endpoints

- `POST /v1/datasets/instagram/upload`
- `POST /v1/datasets/{dataset_id}/build`
- `GET /v1/colab/launch?bundle_id=...`
- `GET /v1/colab/train-script`
- `GET /v1/bundles/{bundle_id}/download?token=...`
- `POST /v1/models/register`

## Git + Privacy Notes

This repo is configured to exclude private artifacts:

- `data/datasets/` (raw/processed message content)
- `data/bundles/` (training bundles)
- `data/models/` (adapter files like `adapter.zip`)
- `.venv/`, caches, pyc files

`.gitignore` ensures message exports and trained artifacts are not committed.

## Test

```powershell
pytest -q
```

## Environment Variables

- `PUBLIC_BASE_URL` use public tunnel URL for Colab-reachable links
- `NOTEBOOK_URL` override notebook URL returned by launch endpoint
- `APP_SECRET` HMAC token secret
- `STORE_RAW_UPLOADS` set `true` only if you intentionally want raw payload snapshots
- `MAX_UPLOAD_MB` per-file upload cap
