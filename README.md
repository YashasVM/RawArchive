# RawArchive

## Intro (What This Actually Does)

RawArchive takes Instagram chat export `.json` files, cleans/parses them, builds a training bundle (`bun_...`), trains a LoRA adapter in Colab, and lets you register and chat with the model (`mdl_...`) locally.

Current supported source: Instagram export JSON.

Architecture-style diagram:

```text
Instagram Export JSON (Current) / WhatsApp / Telegram / Slack / Discord / Google Chat / Signal / iMessage / BlueBubbles / IRC / Microsoft Teams / Matrix / Feishu / LINE / Mattermost / Nextcloud Talk / Nostr / Synology Chat / Tlon / Twitch / Zalo / Zalo Personal / WebChat
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
- Git
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
cd C:\Users\YashasVM\Downloads\code\LLM
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
cd C:\Users\YashasVM\Downloads\code\LLM
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

2. Open website:

- `http://127.0.0.1:8000`

3. In UI:

- Step 1: upload Instagram `.json`
- Step 2: build bundle and copy `bun_...`

### Colab Deploy Steps

Use notebook:

- `colab/train_lora_ultrafast.ipynb`

Steps:

1. Open notebook in Colab.
2. Set runtime GPU to **T4**.
3. Set:
   - `BASE_URL` = public URL of your local API
   - `BUNDLE_ID` = your `bun_...`
4. Run all cells.
5. Download `adapter.zip`.

If you need a public URL:

```powershell
cloudflared tunnel --url http://127.0.0.1:8000
```

Register adapter in UI Step 4:

- Adapter URI: `local://C:/Users/YashasVM/Downloads/code/LLM/data/models/adapter.zip`
- Validation Loss: numeric value
- Style Score: numeric value

## Exact Commands (Run + Upload)

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

Upload only required project files to GitHub:

```powershell
cd C:\Users\YashasVM\Downloads\code\LLM
git status
git add .gitignore README.md app colab scripts tests requirements.txt requirements.inference.txt
git status
git commit -m "Update README structure and deployment guide"
git push -u origin main
```

GitHub repo:

- `https://github.com/YashasVM/RawArchive.git`

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

Made by @Yashas.VM
