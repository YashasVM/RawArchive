# RawArchive

Turn Instagram chat exports into a personalized reply model with FastAPI + Colab LoRA training.

Made by [@yashas.vm](https://github.com/YashasVM)

## 1. What This Project Does

1. Upload one or more Instagram export `.json` files.
2. Parse and normalize chat data.
3. Build a training bundle (`bun_...`).
4. Train a LoRA adapter in Colab.
5. Download `adapter.zip`.
6. Register the adapter as a model (`mdl_...`).
7. Chat locally or in Colab using that style.

## 2. Project Structure

- `app/`: FastAPI backend + local web UI.
- `colab/`: Colab notebooks and training script.
- `scripts/`: local inference chat script.
- `tests/`: API/parser/builder tests.
- `data/`: runtime artifacts (ignored by git).

## 3. Prerequisites

- Windows + PowerShell
- Python `3.11+`
- Git
- Google Colab account
- (Optional) Cloudflare tunnel for Colab to access your local API

## 4. Local Setup (Step by Step)

1. Open terminal in repo root:

```powershell
cd C:\Users\YashasVM\Downloads\code\LLM
```

2. Create and activate virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
pip install -r requirements.txt
```

4. Start the API + website:

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

5. Open:

- `http://127.0.0.1:8000`

## 5. Build Bundle (`bun_...`) From Website

1. In Step 1 UI, upload Instagram `.json` files.
2. Wait until parse stats appear.
3. In Step 2, select the target user.
4. Click **Build Bundle**.
5. Copy and save the returned `bundle_id` (starts with `bun_`).

Made by [@yashas.vm](https://github.com/YashasVM)

## 6. Train Adapter in Colab

Recommended notebook:

- `colab/train_lora_ultrafast.ipynb`

Steps:

1. Open/upload notebook in Google Colab.
2. Set runtime to **T4 GPU**.
3. Fill:
   - `BASE_URL` = your public API URL
   - `BUNDLE_ID` = your `bun_...` id
4. Run all cells in order.
5. Download generated `adapter.zip`.

If you need a tunnel:

```powershell
cloudflared tunnel --url http://127.0.0.1:8000
```

Use the generated `https://...trycloudflare.com` as `BASE_URL`.

## 7. Register Adapter (`mdl_...`) in Website

1. Put downloaded adapter at:

`C:/Users/YashasVM/Downloads/code/LLM/data/models/adapter.zip`

2. In website Step 4, enter:
   - Adapter URI: `local://C:/Users/YashasVM/Downloads/code/LLM/data/models/adapter.zip`
   - Validation Loss: numeric (example `3.83`)
   - Style Score: numeric (example `0.80`)
3. Click **Register Model**.
4. Save the returned `model_id` (`mdl_...`).

## 8. Chat With Your Model

### Option A: Local Chat

1. Install inference dependencies:

```powershell
pip install -r requirements.inference.txt
```

2. Run:

```powershell
python scripts\chat_local.py --model-id mdl_bb8e7abb4c
```

3. Commands:
   - `/reset` clears history
   - `/exit` quits

### Option B: Colab Chat

Notebook:

- `colab/chat_adapter_easy.ipynb`

Steps:

1. Open notebook in Colab.
2. Upload `adapter.zip`.
3. Run all cells.
4. Chat in the last cell.

## 9. API Endpoints

- `POST /v1/datasets/instagram/upload`
- `POST /v1/datasets/{dataset_id}/build`
- `GET /v1/colab/launch?bundle_id=...`
- `GET /v1/colab/train-script`
- `GET /v1/bundles/{bundle_id}/download?token=...`
- `POST /v1/models/register`

## 10. Privacy and Git Safety

The repo ignores private files by default:

- `data/datasets/` (raw message data)
- `data/bundles/` (training bundles)
- `data/models/` (model zips like `adapter.zip`)
- `.venv/`, caches, `*.pyc`
- local-only folders: `.claude/`, `RawArchive/`
- explicit sensitive names: `adapter.zip`, `attachment.zip`, `messages*.json`

This keeps message exports and adapters out of GitHub commits.

## 11. Upload to GitHub (Only Main Files)

Run these commands from repo root:

```powershell
cd C:\Users\YashasVM\Downloads\code\LLM
git status
git add .gitignore README.md app colab scripts tests requirements.txt requirements.inference.txt
git status
git commit -m "Clean README and tighten privacy ignore rules"
git push -u origin main
```

After pushing, confirm no private files are tracked:

```powershell
git ls-files | Select-String -Pattern "adapter\.zip|attachment\.zip|messages?|conversation"
```

The expected output is empty.

## 12. Run Tests

```powershell
pytest -q
```

## 13. Environment Variables

- `PUBLIC_BASE_URL`: public URL for Colab callbacks/download links
- `NOTEBOOK_URL`: custom notebook URL returned by launch endpoint
- `APP_SECRET`: HMAC token secret
- `STORE_RAW_UPLOADS`: set `true` only if you intentionally want raw payload snapshots
- `MAX_UPLOAD_MB`: upload size limit per file
