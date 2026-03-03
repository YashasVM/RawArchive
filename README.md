# RawArchive

## Intro (What This Actually Does)

RawArchive is an end-to-end training and inference app for building a personalized chat reply model from your exported messages.

You run the app locally on your PC for:

- data upload
- parsing and dataset preparation
- model registration
- local chat inference

You use Google Colab for GPU training to fine-tune `Qwen/Qwen2.5-3B-Instruct` with LoRA on your bundle data.

In practical terms:

1. You upload Instagram export `.json`.
2. RawArchive converts those conversations into supervised training samples.
3. Colab trains a LoRA adapter on top of Qwen 2.5 3B.
4. You register the generated `adapter.zip`.
5. You chat with responses that follow the learned writing style from your message history.

Current supported source: Instagram export JSON.

## Detailed Execution Diagram

```mermaid
flowchart TD
    A[Instagram Export JSON] --> B[Web UI Upload<br/>app/static/index.html]
    B --> C[FastAPI Upload Endpoint<br/>POST /v1/datasets/instagram/upload]
    C --> D[Parser + Normalizer<br/>app/parser.py]
    D --> E[Dataset Builder<br/>app/dataset_builder.py]
    E --> F[Bundle Stored<br/>data/bundles/bun_*]

    F --> G[Colab Notebook<br/>colab/train_lora_ultrafast.ipynb]
    G --> H[Download Bundle via Signed URL<br/>GET /v1/bundles/{bundle_id}/download]
    H --> I[Training Data in Colab Runtime]

    J[Qwen/Qwen2.5-3B-Instruct<br/>Base Model Weights] --> K[LoRA Fine-Tuning<br/>colab/train_lora.py]
    I --> K
    K --> L[adapter.zip]

    L --> M[Model Registration<br/>POST /v1/models/register]
    M --> N[Model ID Created<br/>mdl_*]
    N --> O[Inference Runtime<br/>scripts/chat_local.py]
    O --> P[Style-Matched Response]
```

Execution boundary:

- Local machine: upload, parse, bundle, registry, local inference.
- Colab GPU runtime: Qwen base model load + LoRA training loop.

## How It Works Internally (Step by Step)

### Stage 1: Data Ingestion

1. The UI sends your Instagram export files to `POST /v1/datasets/instagram/upload`.
2. The parser extracts conversation events, sender names, timestamps, and text.
3. The normalizer cleans formatting and filters unusable records.

Output of this stage:

- structured dataset metadata
- parsed conversation blocks for training preparation

### Stage 2: Training Sample Construction

1. You choose the target sender/style in the UI.
2. The builder creates prompt-response style pairs.
3. Samples are split into train and validation sets.
4. RawArchive writes a bundle with an ID like `bun_xxxxx`.

Output of this stage:

- reusable bundle artifact under `data/bundles/`
- `bundle_id` used by Colab training

### Stage 3: Colab Training Pipeline

1. Colab notebook receives:
   - `BASE_URL` pointing to your local API via Cloudflare
   - `BUNDLE_ID` for the dataset
2. Notebook downloads the signed bundle.
3. `Qwen/Qwen2.5-3B-Instruct` base model is loaded in Colab.
4. LoRA layers are attached and only adapter params are trained.
5. Training and validation execute for configured steps.
6. Adapter is exported as `adapter.zip`.

Output of this stage:

- LoRA adapter checkpoint zip ready for registration

### Stage 4: Registration and Inference

1. You provide adapter URI in UI Step 4.
2. API registers model metadata and assigns `mdl_*`.
3. Local chat script loads base model + registered adapter.
4. Inference produces responses in the learned conversation style.

Output of this stage:

- interactive model chat using your trained style

## What Are All The Requirements

System:

- Windows + PowerShell
- Python `3.11+`
- Google Colab account
- Internet access for model downloads in Colab
- Optional but recommended: `cloudflared` for local-to-Colab connectivity

Python dependencies:

- API/UI/runtime deps: `requirements.txt`
- local inference deps: `requirements.inference.txt`

Project paths used by workflow:

- API entrypoint: `app/main.py`
- parser: `app/parser.py`
- dataset builder: `app/dataset_builder.py`
- training notebook: `colab/train_lora_ultrafast.ipynb`
- training script: `colab/train_lora.py`
- local chat script: `scripts/chat_local.py`

## Deploy (Local + Colab)

### Local Deploy Steps

1. Create environment and install API dependencies:

```powershell
cd C:\Users\{your-username}\Downloads\code\LLM
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Start API server:

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

3. In a second terminal, start Cloudflare tunnel for Colab:

```powershell
cd C:\Users\{your-username}\Downloads\code\LLM
.\.venv\Scripts\Activate.ps1
cloudflared tunnel --url http://127.0.0.1:8000
```

4. Open the UI:

- `http://127.0.0.1:8000`

5. Upload `.json` and build bundle in UI:

- Step 1: upload Instagram export files
- Step 2: build bundle and copy `bun_*`

### Colab Deploy Steps

Notebook:

- `colab/train_lora_ultrafast.ipynb`

Steps:

1. Open notebook in Colab.
2. Set runtime to **T4 GPU**.
3. Set notebook variables:
   - `BASE_URL` = your Cloudflare URL `https://...trycloudflare.com`
   - `BUNDLE_ID` = the `bun_*` from local UI
4. Run all cells to train LoRA on Qwen 2.5 3B.
5. Download resulting `adapter.zip`.

Register adapter in local UI Step 4:

- Adapter URI: `local://C:/Users/{your-username}/Downloads/code/LLM/data/models/adapter.zip`
- Validation Loss: numeric value
- Style Score: numeric value

## Exact Commands (Run)

Run API:

```powershell
cd C:\Users\{your-username}\Downloads\code\LLM
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Run Cloudflare tunnel:

```powershell
cd C:\Users\{your-username}\Downloads\code\LLM
.\.venv\Scripts\Activate.ps1
cloudflared tunnel --url http://127.0.0.1:8000
```

Run local inference chat:

```powershell
cd C:\Users\{your-username}\Downloads\code\LLM
.\.venv\Scripts\Activate.ps1
pip install -r requirements.inference.txt
python scripts\chat_local.py --model-id mdl_your_model_id
```

Run tests:

```powershell
cd C:\Users\{your-username}\Downloads\code\LLM
.\.venv\Scripts\Activate.ps1
pytest -q
```

## Component-Level Execution Order

1. `app/main.py` boots FastAPI and serves endpoints/UI.
2. `app/parser.py` parses uploaded Instagram JSON.
3. `app/dataset_builder.py` builds training examples and bundles.
4. `colab/train_lora_ultrafast.ipynb` orchestrates Colab training.
5. `colab/train_lora.py` runs the actual LoRA fine-tuning loop.
6. UI/API registers `adapter.zip` to create `mdl_*`.
7. `scripts/chat_local.py` loads model + adapter for chat inference.

## Privacy Check (Exclude Adapter + Messages)

Ignored by git:

- `data/models/adapter.zip`
- `data/datasets/` (raw message content)
- `data/bundles/` (training artifacts)
- patterns like `messages*.json` and `conversation*.json`

Verify no private artifacts are tracked:

```powershell
git ls-files | Select-String -Pattern "adapter\.zip|attachment\.zip|messages?\.json|conversation.*\.json"
```

Expected result: no output.

## Troubleshooting Quick Notes

- Colab cannot access `localhost` directly.
- Always use Cloudflare tunnel URL as `BASE_URL`.
- Keep local API running while Colab downloads bundle/train script.
- If model registration fails, verify adapter path and URI prefix `local://`.

<p align="center">Made by @Yashas.VM</p>
