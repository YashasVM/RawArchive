from __future__ import annotations

import io
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from app.config import (
    APP_SECRET,
    DEFAULT_BASE_MODEL,
    DEFAULT_NOTEBOOK_URL,
    MAX_UPLOAD_MB,
    PUBLIC_BASE_URL,
    STORE_RAW_UPLOADS,
)
from app.dataset_builder import build_training_examples, compute_dataset_stats
from app.models import (
    BuildRequest,
    BuildResponse,
    ColabLaunchResponse,
    RegisterModelRequest,
    RegisterModelResponse,
    UploadResponse,
)
from app.parser import parse_instagram_export
from app.storage import LocalStore
from app.token_utils import create_download_token, verify_download_token

app = FastAPI(title="Instagram Chat Style Trainer", version="0.1.0")
store = LocalStore()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _resolve_base_url(http_request: Request) -> str:
    if PUBLIC_BASE_URL:
        return PUBLIC_BASE_URL
    return str(http_request.base_url).rstrip("/")


@app.get("/v1/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/datasets/instagram/upload", response_model=UploadResponse)
async def upload_instagram_json(
    files: list[UploadFile] | None = File(default=None),
    file: UploadFile | None = File(default=None),
) -> UploadResponse:
    upload_files = list(files or [])
    if file is not None:
        upload_files.append(file)
    if not upload_files:
        raise HTTPException(status_code=400, detail="Upload at least one .json file")

    all_messages = []
    participant_ids: set[str] = set()
    warning_set: set[str] = set()
    raw_payloads = [] if STORE_RAW_UPLOADS else None
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024

    for index, upload_file in enumerate(upload_files):
        filename = upload_file.filename or f"upload_{index}.json"
        if not filename.lower().endswith(".json"):
            raise HTTPException(status_code=400, detail=f"Only .json files are supported. Invalid file: {filename}")

        content = await upload_file.read()
        if len(content) > max_bytes:
            raise HTTPException(status_code=413, detail=f"{filename} exceeds {MAX_UPLOAD_MB}MB limit")

        try:
            payload = json.loads(content.decode("utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {filename}") from exc

        try:
            parse_result = parse_instagram_export(payload, source_tag=filename)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"{filename}: {exc}") from exc

        all_messages.extend(parse_result.messages)
        participant_ids.update(parse_result.participants)
        warning_set.update(parse_result.warnings)
        if raw_payloads is not None:
            raw_payloads.append({"filename": filename, "payload": payload})

    if not all_messages:
        raise HTTPException(status_code=400, detail="No text messages found across uploaded files")

    stats = compute_dataset_stats(all_messages)
    dataset_id = store.create_dataset(
        raw_payload={"files": raw_payloads} if raw_payloads is not None else None,
        messages=all_messages,
        participants=sorted(participant_ids),
        warnings=sorted(warning_set),
        stats=stats,
    )

    return UploadResponse(
        dataset_id=dataset_id,
        stats=stats,
        warnings=sorted(warning_set),
        participants=sorted(participant_ids),
    )


@app.post("/v1/datasets/{dataset_id}/build", response_model=BuildResponse)
def build_dataset_bundle(dataset_id: str, request: BuildRequest, http_request: Request) -> BuildResponse:
    dataset = store.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    participant_set = set(dataset["meta"].get("participants", []))
    if request.target_user_id not in participant_set:
        raise HTTPException(status_code=400, detail="target_user_id is not in dataset participants")

    train_examples, val_examples, meta = build_training_examples(
        messages=dataset["messages"],
        target_user_id=request.target_user_id,
        context_turns=request.context_turns,
        min_reply_chars=request.min_reply_chars,
        max_samples=request.max_samples,
        val_ratio=request.val_ratio,
    )

    if not train_examples:
        raise HTTPException(
            status_code=400,
            detail="Not enough target-user text samples after filtering. Lower min_reply_chars or upload richer data.",
        )

    train_config = {
        "base_model": DEFAULT_BASE_MODEL,
        "lora": {"r": 16, "alpha": 32, "dropout": 0.05},
        "seq_len": 512,
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "epochs": 1,
        "learning_rate": 2e-4,
        "scheduler": "cosine",
        "use_4bit": True,
    }

    bundle_id = store.create_bundle(
        dataset_id=dataset_id,
        train_examples=train_examples,
        val_examples=val_examples,
        dataset_meta=meta,
        train_config=train_config,
    )

    token = create_download_token(bundle_id=bundle_id, secret=APP_SECRET)
    base = _resolve_base_url(http_request)
    download_url = f"{base}/v1/bundles/{bundle_id}/download?token={token}"

    return BuildResponse(
        bundle_id=bundle_id,
        train_examples=len(train_examples),
        val_examples=len(val_examples),
        download_url=download_url,
    )


@app.get("/v1/colab/launch", response_model=ColabLaunchResponse)
def colab_launch(bundle_id: str, http_request: Request) -> ColabLaunchResponse:
    manifest = store.get_bundle_manifest(bundle_id)
    if not manifest:
        raise HTTPException(status_code=404, detail="Bundle not found")

    token = create_download_token(bundle_id=bundle_id, secret=APP_SECRET)
    base = _resolve_base_url(http_request)
    download_url = f"{base}/v1/bundles/{bundle_id}/download?token={token}"

    return ColabLaunchResponse(
        notebook_url=DEFAULT_NOTEBOOK_URL,
        env={
            "BUNDLE_URL": download_url,
            "BASE_MODEL": DEFAULT_BASE_MODEL,
            "BUNDLE_ID": bundle_id,
        },
    )


@app.get("/v1/colab/train-script")
def colab_train_script() -> FileResponse:
    script_path = Path(__file__).resolve().parent.parent / "colab" / "train_lora.py"
    if not script_path.exists():
        raise HTTPException(status_code=404, detail="Training script not found")
    return FileResponse(script_path, media_type="text/x-python", filename="train_lora.py")


@app.get("/v1/bundles/{bundle_id}/download")
def download_bundle(bundle_id: str, token: str = Query(..., min_length=10)) -> Response:
    if not verify_download_token(bundle_id=bundle_id, token=token, secret=APP_SECRET):
        raise HTTPException(status_code=403, detail="Invalid or expired token")

    bundle_dir = store.get_bundle_dir(bundle_id)
    if not bundle_dir:
        raise HTTPException(status_code=404, detail="Bundle not found")

    required_files = [
        "train.jsonl",
        "val.jsonl",
        "dataset_meta.json",
        "train_config.yaml",
        "manifest.json",
    ]

    memory_file = io.BytesIO()
    with ZipFile(memory_file, mode="w", compression=ZIP_DEFLATED) as archive:
        for filename in required_files:
            full_path = bundle_dir / filename
            if not full_path.exists():
                raise HTTPException(status_code=500, detail=f"Bundle file missing: {filename}")
            archive.writestr(filename, full_path.read_bytes())

    headers = {"Content-Disposition": f'attachment; filename="{bundle_id}.zip"'}
    return Response(content=memory_file.getvalue(), media_type="application/zip", headers=headers)


@app.post("/v1/models/register", response_model=RegisterModelResponse)
def register_model(request: RegisterModelRequest) -> RegisterModelResponse:
    if not store.get_bundle_manifest(request.bundle_id):
        raise HTTPException(status_code=404, detail="Bundle not found")

    model_id = store.register_model(
        bundle_id=request.bundle_id,
        adapter_uri=request.adapter_uri,
        base_model=request.base_model,
        metrics=request.metrics,
    )
    return RegisterModelResponse(model_id=model_id, status="registered")


app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


@app.get("/")
def root() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")
