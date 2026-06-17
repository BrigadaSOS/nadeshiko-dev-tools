"""Modal GPU backend for the NSFW tagger.

Imported lazily by `tag-media --modal`. Importing this module requires the
optional `modal` dependency (`uv sync --extra modal`); the CLI catches the
ImportError and prints an install hint when it is missing.

Each target folder's `_data.json` + the screenshots still needing a rating are
packed into a tar.gz, tagged on a Modal T4 GPU, and written back in place after a
timestamped backup. Works for both anime episode folders and YouTube video-ID
folders — callers pass the already-discovered (label, path) targets.
"""

from __future__ import annotations

import io
import json
import shutil
import tarfile
import time
from pathlib import Path
from typing import Any

import modal
from rich.console import Console

from nadeshiko_dev_tools.nsfw_tagger.classifier import (
    KAOMOJIS,
    MODEL_REPO,
    RATING_MAP,
    TAG_THRESHOLD,
)

console = Console()

APP_NAME = "nadeshiko-nsfw-tagger"

app = modal.App(APP_NAME)
image = modal.Image.from_registry(
    "nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04",
    add_python="3.11",
).pip_install(
    "huggingface-hub>=0.20.0",
    "numpy>=1.24.0",
    "onnxruntime-gpu>=1.20.0",
    "pandas>=2.0.0",
    "Pillow>=10.0.0",
)


def _status(ep_dir: Path) -> dict[str, int]:
    data = json.loads((ep_dir / "_data.json").read_text())
    segs = data.get("segments", [])
    return {
        "segments": len(segs),
        "tagged": sum(1 for s in segs if s.get("content_rating") is not None),
        "analysis": sum(1 for s in segs if s.get("content_analysis") is not None),
        "screenshots": sum(1 for s in segs if s.get("files", {}).get("screenshot")),
    }


def _make_payload(ep_dir: Path, force: bool = False) -> bytes:
    data = json.loads((ep_dir / "_data.json").read_text())
    needed: set[str] = set()
    for seg in data.get("segments", []):
        if force:
            seg["content_rating"] = None
            seg["content_analysis"] = None
        screenshot = seg.get("files", {}).get("screenshot")
        if screenshot and (force or seg.get("content_analysis") is None):
            needed.add(screenshot)

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        encoded = json.dumps(data, ensure_ascii=False).encode("utf-8")
        info = tarfile.TarInfo("_data.json")
        info.size = len(encoded)
        tf.addfile(info, io.BytesIO(encoded))
        for name in sorted(needed):
            path = ep_dir / name
            if path.exists():
                tf.add(path, arcname=name)
    return buf.getvalue()


@app.function(gpu="T4", image=image, timeout=1800, memory=16384, cpu=4)
def tag_episode(payload: bytes, batch_size: int = 32) -> dict[str, Any]:
    import dataclasses
    import subprocess
    import tempfile
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    import onnxruntime as rt
    import pandas as pd
    from huggingface_hub import hf_hub_download
    from PIL import Image

    started = time.time()
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()

    @dataclasses.dataclass
    class ClassificationResult:
        rating: str
        rating_scores: dict[str, float]
        content_rating: str
        tags: dict[str, float]

    class WDTagger:
        def __init__(self, model_repo: str = MODEL_REPO):
            csv_path = hf_hub_download(model_repo, "selected_tags.csv")
            model_path = hf_hub_download(model_repo, "model.onnx")
            df = pd.read_csv(csv_path)
            self.tag_names = (
                df["name"].apply(lambda x: x.replace("_", " ") if x not in KAOMOJIS else x).tolist()
            )
            self.rating_indexes = list(np.where(df["category"] == 9)[0])
            self.general_indexes = list(np.where(df["category"] == 0)[0])
            session_options = rt.SessionOptions()
            session_options.enable_mem_pattern = False
            self.model = rt.InferenceSession(
                model_path,
                sess_options=session_options,
                providers=[
                    ("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"})
                ],
            )
            self.active_provider = self.model.get_providers()[0]
            _, self.target_size, _, _ = self.model.get_inputs()[0].shape
            self.input_name = self.model.get_inputs()[0].name
            self.output_name = self.model.get_outputs()[0].name

        def _prepare_image(self, image_path: str | Path) -> np.ndarray:
            image = Image.open(image_path).convert("RGBA")
            canvas = Image.new("RGBA", image.size, (255, 255, 255))
            canvas.alpha_composite(image)
            image = canvas.convert("RGB")
            w, h = image.size
            max_dim = max(w, h)
            padded = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
            padded.paste(image, ((max_dim - w) // 2, (max_dim - h) // 2))
            if max_dim != self.target_size:
                padded = padded.resize((self.target_size, self.target_size), Image.BICUBIC)
            arr = np.asarray(padded, dtype=np.float32)
            return np.expand_dims(arr[:, :, ::-1], axis=0)

        def classify_batch(self, image_paths: list[str | Path]) -> list[ClassificationResult]:
            workers = min(4, len(image_paths)) or 1
            with ThreadPoolExecutor(max_workers=workers) as pool:
                arrays = list(pool.map(lambda p: self._prepare_image(p)[0], image_paths))
            preds = self.model.run([self.output_name], {self.input_name: np.stack(arrays, axis=0)})[
                0
            ]
            return [self._parse_predictions(preds[i]) for i in range(len(image_paths))]

        def _parse_predictions(self, pred: np.ndarray) -> ClassificationResult:
            labels = list(zip(self.tag_names, pred.astype(float), strict=True))
            rating_scores = {labels[i][0]: labels[i][1] for i in self.rating_indexes}
            top_rating = max(rating_scores, key=rating_scores.get)
            tags = {}
            for i in self.general_indexes:
                tag_name, score = labels[i]
                if score >= TAG_THRESHOLD:
                    tags[tag_name] = score
            return ClassificationResult(
                rating=top_rating,
                rating_scores={k: round(v, 4) for k, v in rating_scores.items()},
                content_rating=RATING_MAP.get(top_rating, "SAFE"),
                tags={k: round(v, 4) for k, v in sorted(tags.items(), key=lambda x: -x[1])},
            )

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as tf:
            tf.extractall(root)
        data = json.loads((root / "_data.json").read_text())
        segments = data.get("segments", [])
        to_tag = []
        for i, seg in enumerate(segments):
            screenshot = seg.get("files", {}).get("screenshot")
            if screenshot and (root / screenshot).exists() and seg.get("content_analysis") is None:
                to_tag.append((i, root / screenshot))

        load_started = time.time()
        tagger = WDTagger()
        load_seconds = time.time() - load_started
        inference_started = time.time()
        for chunk_start in range(0, len(to_tag), batch_size):
            chunk = to_tag[chunk_start : chunk_start + batch_size]
            results = tagger.classify_batch([path for _, path in chunk])
            for (seg_idx, _), result in zip(chunk, results, strict=True):
                segments[seg_idx]["content_rating"] = result.content_rating
                segments[seg_idx]["content_analysis"] = {
                    "scores": result.rating_scores,
                    "tags": result.tags,
                }
        encoded = json.dumps(data, ensure_ascii=False).encode("utf-8")
        return {
            "gpu": gpu,
            "provider": tagger.active_provider,
            "segments": len(segments),
            "tagged": sum(1 for s in segments if s.get("content_rating") is not None),
            "load_seconds": round(load_seconds, 3),
            "inference_seconds": round(time.time() - inference_started, 3),
            "total_remote_seconds": round(time.time() - started, 3),
            "data_json": encoded.decode("utf-8"),
        }


def run_modal(
    targets: list[tuple[str, str]],
    batch_size: int = 32,
    force: bool = False,
) -> None:
    """Tag each target folder on Modal GPU, writing _data.json back in place.

    targets: (label, dir_path) pairs as discovered by the CLI. Raises on any
    Modal failure so the caller can fall back to local execution.
    """
    console.print(f"[cyan bold]Tagging on Modal GPU ({APP_NAME})...[/cyan bold]")
    with app.run():
        for label, dir_path in targets:
            ep_dir = Path(dir_path)
            before = _status(ep_dir)
            if before["segments"] == before["analysis"] and not force:
                console.print(f"  {label}: Already tagged ({before['segments']} segments)")
                continue

            console.print(f"  {label}: sending {before['screenshots']} screenshots to Modal...")
            payload = _make_payload(ep_dir, force=force)
            result = tag_episode.remote(payload, batch_size=batch_size)

            data_json = result.pop("data_json")
            data_path = ep_dir / "_data.json"
            backup = ep_dir / f"_data.pre-modal-{time.strftime('%Y%m%d-%H%M%S')}.json"
            shutil.copy2(data_path, backup)
            tmp = ep_dir / "_data.json.modal.tmp"
            tmp.write_text(data_json, encoding="utf-8")
            tmp.replace(data_path)

            console.print(
                f"  {label}: Done — {result['tagged']}/{result['segments']} tagged on "
                f"{result['provider']} in {result['inference_seconds']}s "
                f"(backup: {backup.name})"
            )

    console.print("[green bold]Modal tagger complete[/green bold]")
