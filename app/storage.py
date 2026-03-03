from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import yaml

from app.config import BUNDLES_DIR, DATASETS_DIR, MODELS_DIR
from app.models import CanonicalMessage, TrainingExample


class LocalStore:
    def __init__(self) -> None:
        self.datasets_dir = DATASETS_DIR
        self.bundles_dir = BUNDLES_DIR
        self.models_dir = MODELS_DIR

    def _new_id(self, prefix: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex[:10]}"

    def _write_json(self, path: Path, payload: Any) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _read_json(self, path: Path) -> Any:
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_jsonl(self, path: Path, payload: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as file:
            for item in payload:
                file.write(json.dumps(item, ensure_ascii=False))
                file.write("\n")

    def _read_jsonl(self, path: Path) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items

    def create_dataset(
        self,
        raw_payload: Any | None,
        messages: list[CanonicalMessage],
        participants: list[str],
        warnings: list[str],
        stats: dict[str, Any],
    ) -> str:
        dataset_id = self._new_id("ds")
        dataset_dir = self.datasets_dir / dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True)

        if raw_payload is not None:
            self._write_json(dataset_dir / "raw.json", raw_payload)
        self._write_jsonl(dataset_dir / "canonical.jsonl", [msg.model_dump() for msg in messages])
        self._write_json(
            dataset_dir / "meta.json",
            {
                "dataset_id": dataset_id,
                "participants": participants,
                "warnings": warnings,
                "stats": stats,
            },
        )
        return dataset_id

    def get_dataset(self, dataset_id: str) -> dict[str, Any] | None:
        dataset_dir = self.datasets_dir / dataset_id
        if not dataset_dir.exists():
            return None

        meta = self._read_json(dataset_dir / "meta.json")
        canonical_raw = self._read_jsonl(dataset_dir / "canonical.jsonl")
        messages = [CanonicalMessage(**item) for item in canonical_raw]
        return {
            "meta": meta,
            "messages": messages,
            "raw_path": dataset_dir / "raw.json" if (dataset_dir / "raw.json").exists() else None,
            "dataset_dir": dataset_dir,
        }

    def create_bundle(
        self,
        dataset_id: str,
        train_examples: list[TrainingExample],
        val_examples: list[TrainingExample],
        dataset_meta: dict[str, Any],
        train_config: dict[str, Any],
    ) -> str:
        bundle_id = self._new_id("bun")
        bundle_dir = self.bundles_dir / bundle_id
        bundle_dir.mkdir(parents=True, exist_ok=True)

        self._write_jsonl(
            bundle_dir / "train.jsonl",
            [{"input": item.input_text, "output": item.target_text} for item in train_examples],
        )
        self._write_jsonl(
            bundle_dir / "val.jsonl",
            [{"input": item.input_text, "output": item.target_text} for item in val_examples],
        )
        self._write_json(bundle_dir / "dataset_meta.json", dataset_meta)
        (bundle_dir / "train_config.yaml").write_text(
            yaml.safe_dump(train_config, sort_keys=False),
            encoding="utf-8",
        )
        self._write_json(
            bundle_dir / "manifest.json",
            {
                "bundle_id": bundle_id,
                "dataset_id": dataset_id,
                "train_examples": len(train_examples),
                "val_examples": len(val_examples),
            },
        )
        return bundle_id

    def get_bundle_dir(self, bundle_id: str) -> Path | None:
        bundle_dir = self.bundles_dir / bundle_id
        return bundle_dir if bundle_dir.exists() else None

    def get_bundle_manifest(self, bundle_id: str) -> dict[str, Any] | None:
        bundle_dir = self.get_bundle_dir(bundle_id)
        if not bundle_dir:
            return None
        return self._read_json(bundle_dir / "manifest.json")

    def register_model(
        self,
        bundle_id: str,
        adapter_uri: str,
        base_model: str,
        metrics: dict[str, Any],
    ) -> str:
        model_id = self._new_id("mdl")
        payload = {
            "model_id": model_id,
            "bundle_id": bundle_id,
            "adapter_uri": adapter_uri,
            "base_model": base_model,
            "metrics": metrics,
        }
        self._write_json(self.models_dir / f"{model_id}.json", payload)
        return model_id
