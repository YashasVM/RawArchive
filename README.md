# Instagram Chat Style Trainer (Colab-First MVP)

This project lets users upload one or many Instagram chat export files (`.json`), build a personalized training bundle, and fine-tune a LoRA adapter on `Qwen/Qwen2.5-3B-Instruct` using Google Colab.

## What is implemented

- FastAPI backend with endpoints for:
  - `POST /v1/datasets/instagram/upload`
  - `POST /v1/datasets/{dataset_id}/build`
  - `GET /v1/colab/launch?bundle_id=...`
  - `GET /v1/colab/train-script`
  - `GET /v1/bundles/{bundle_id}/download?token=...`
  - `POST /v1/models/register`
- Instagram parser with canonical message normalization
- Training example builder (context -> next target-user reply)
- Local artifact store for datasets/bundles/models
- Signed bundle download tokens
- Web UI (`/`) for upload/build/launch/register flow
- Colab assets (`colab/train_lora.py`, `colab/train_lora.ipynb`, `colab/train_lora_easy.ipynb`)
- Unit + integration tests

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`.

## Colab workflow

1. Upload a chat JSON in the web UI.
2. Build a bundle for a selected target user.
3. Use `GET /v1/colab/launch?bundle_id=...` output:
   - `env.BUNDLE_URL` is the signed bundle URL.
   - `notebook_url` points to your Colab notebook location.
4. In Colab, run `colab/train_lora.ipynb` or execute:

```bash
python -m colab.train_lora --bundle-url "$BUNDLE_URL" --base-model "Qwen/Qwen2.5-3B-Instruct"
```

5. Register the adapter URI with `POST /v1/models/register`.

### Easier Colab flow (Base URL + Bundle ID only)

Use `colab/train_lora_easy.ipynb`:
1. Open it in Colab.
2. Enter `BASE_URL` (your public tunnel URL) and `BUNDLE_ID`.
3. The notebook auto-fetches launch config + `train_lora.py`.
4. Run training and download `adapter.zip`.

## Local chat inference

After downloading `adapter.zip` and registering your model, run a local chat loop:

```bash
pip install -r requirements.inference.txt
python scripts/chat_local.py --model-id mdl_bb8e7abb4c
```

Optional flags:
- `--system-prompt "..."` to tune style behavior
- `--temperature 0.7 --top-p 0.9`
- `--no-4bit` if 4-bit loading is not available in your environment

Chat commands:
- `/reset` clears conversation history
- `/exit` quits

## Notes

- Raw chat data is stored locally under `data/` in this MVP.
- `NOTEBOOK_URL` can be overridden by env var for your hosted notebook link.
- Set `PUBLIC_BASE_URL` (for example your Cloudflare tunnel URL) so generated `BUNDLE_URL` is reachable from Colab even when you use `localhost` UI for fast uploads.
- `STORE_RAW_UPLOADS=false` by default to keep multi-file uploads faster. Set it to `true` only if you need raw payload snapshots under `data/datasets/.../raw.json`.
- `APP_SECRET` should be changed in production.
