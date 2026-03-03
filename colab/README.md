# Colab Training

## Option 0: Fast inference chat (use trained adapter)

1. Open `chat_adapter_easy.ipynb` in Colab.
2. Upload your downloaded `adapter.zip`.
3. Run all cells and chat in the last cell.

## Option 1 (Recommended): Easy notebook

1. Open `train_lora_easy.ipynb` in Colab.
2. Enter:
   - `BASE_URL` (public tunnel URL, e.g. Cloudflare/ngrok)
   - `BUNDLE_ID` (`bun_...`)
3. Run remaining cells. The notebook auto-fetches `BUNDLE_URL` and training script.
4. Download `adapter.zip` and register URI via `/v1/models/register`.

## Option 2: Manual notebook

1. Get `BUNDLE_URL` from `/v1/colab/launch?bundle_id=...`.
2. Open `train_lora.ipynb` in Colab.
3. Set `BUNDLE_URL` and run all cells.
4. Download `outputs/adapter` and register the URI via `/v1/models/register`.
