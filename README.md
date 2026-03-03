# RawArchive

## Intro (What This Actually Does)

RawArchive is an end-to-end app that you can run on your PC and connect to Google Colab for training. It ingests your Instagram chat export `.json` files, preprocesses the conversations into supervised training pairs, and fine-tunes `Qwen/Qwen2.5-3B-Instruct` using a LoRA adapter.

After training, the adapter is registered in the app so you can chat with a model that responds in the learned style of the message history. In short: local app for data + control, Colab for GPU training, and a personalized response model for inference.

Current supported source: Instagram export JSON.

Architecture-style diagram:

```text
    Instagram Export JSON (Current) 
               |
               v
+-------------------------------+
|         RawArchive API        |
|       (control plane)         |
|      http://127.0.0.1:8000    |
+--------------+----------------+
               |
               |- Bundle Builder (`bun_*`)
               |- Colab Trainer (`colab/*.ipynb`)
               |- Web UI (`app/static/index.html`)
               |- Model Registry (`mdl_*`)
               '- Local Chat (`scripts/chat_local.py`)
```

## What Are All The Requirements

System:

- Windows + PowerShell
- Python `3.11+`
- Google Colab account
- Optional: `cloudflared` tunnel for Colab -> local API access

Project files required:

- `app/`
- `colab/`
- `scripts/`
- `tests/`
- `requirements.txt`
- `requirements.inference.txt`

Install dependencies:

```powershell
cd C:\Users\{your-username}\Downloads\code\LLM
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Inference dependencies (for local chat):

```powershell
pip install -r requirements.inference.txt
```

## Deploy (Local + Colab)

### Local Deploy Steps

1. Start backend + website:

```powershell
cd C:\Users\{your-username}\Downloads\code\LLM
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

2. Start Cloudflare tunnel in a second terminal (required for Colab access):

```powershell
cd C:\Users\{your-username}\Downloads\code\LLM
.\.venv\Scripts\Activate.ps1
cloudflared tunnel --url http://127.0.0.1:8000
```

3. Open website:

- `http://127.0.0.1:8000`

4. In UI:

- Step 1: upload Instagram `.json`
- Step 2: build bundle and copy `bun_...`

### Colab Deploy Steps

Use notebook:

- `colab/train_lora_ultrafast.ipynb`

Steps:

1. Open notebook in Colab.
2. Set runtime GPU to **T4**.
3. Set:
   - `BASE_URL` = Cloudflare URL from local tunnel (`https://...trycloudflare.com`)
   - `BUNDLE_ID` = your `bun_...`
4. Run all cells.
5. Download `adapter.zip`.

Register adapter in UI Step 4:

- Adapter URI: `local://C:/Users/{your-username}/Downloads/code/LLM/data/models/adapter.zip`
- Validation Loss: numeric value
- Style Score: numeric value

## Exact Commands (Run)

Run API:

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Run local chat with model:

```powershell
python scripts\chat_local.py --model-id mdl_your_model_id
```

Run tests:

```powershell
pytest -q
```

## Privacy Check (Exclude adapter + messages)

Ignored by git:

- `data/models/adapter.zip`
- `data/datasets/` (message data)
- `data/bundles/`
- message export patterns (`messages*.json`, `conversation*.json`)

Verify nothing private is tracked:

```powershell
git ls-files | Select-String -Pattern "adapter\.zip|attachment\.zip|messages?\.json|conversation.*\.json"
```

Expected result: no output.

<p align="center">Made by @Yashas.VM</p>
