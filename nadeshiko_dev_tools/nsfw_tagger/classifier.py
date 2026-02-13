"""Content rating classifier using WaifuDiffusion Tagger v3.

Uses the SwinV2 variant (wd-swinv2-tagger-v3), a Danbooru tag predictor that
outputs ratings (general/sensitive/questionable/explicit) and content tags.
Maps Danbooru ratings to backend content ratings (SAFE/SUGGESTIVE/QUESTIONABLE/EXPLICIT).
"""

import ctypes
import os
from ctypes.util import find_library
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as rt
import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image

MODEL_REPO = "SmilingWolf/wd-swinv2-tagger-v3"

TAG_THRESHOLD = 0.25

# Tags where underscores should NOT be replaced with spaces
KAOMOJIS = {
    "0_0", "(o)_(o)", "+_+", "+_-", "._.", "<o>_<o>", "<|>_<|>",
    "=_=", ">_<", "3_3", "6_9", ">_o", "@_@", "^_^", "o_o",
    "u_u", "x_x", "|_|", "||_||",
}

# Danbooru rating -> backend content rating (uppercase to match backend enum)
RATING_MAP = {
    "general": "SAFE",
    "sensitive": "SUGGESTIVE",
    "questionable": "QUESTIONABLE",
    "explicit": "EXPLICIT",
}


@dataclass
class ClassificationResult:
    """Result of classifying a single image."""

    rating: str  # Danbooru rating: general, sensitive, questionable, explicit
    rating_scores: dict[str, float]
    content_rating: str  # Backend enum: SAFE, SUGGESTIVE, QUESTIONABLE, EXPLICIT
    tags: dict[str, float]  # All general tags above threshold


class WDTagger:
    """WaifuDiffusion Tagger v3 wrapper for content rating classification."""

    def __init__(self, model_repo: str = MODEL_REPO):
        csv_path = hf_hub_download(model_repo, "selected_tags.csv")
        model_path = hf_hub_download(model_repo, "model.onnx")

        df = pd.read_csv(csv_path)
        self.tag_names = df["name"].apply(
            lambda x: x.replace("_", " ") if x not in KAOMOJIS else x
        ).tolist()

        self.rating_indexes = list(np.where(df["category"] == 9)[0])
        self.general_indexes = list(np.where(df["category"] == 0)[0])

        require_gpu = os.getenv("NSFW_TAGGER_REQUIRE_GPU", "").strip().lower() in {
            "1", "true", "yes", "on",
        }
        cuda_ready = self._can_use_cuda_provider()

        if require_gpu and not cuda_ready:
            raise RuntimeError(
                "NSFW_TAGGER_REQUIRE_GPU is set, but CUDAExecutionProvider is unavailable. "
                "Install the CUDA-matching onnxruntime-gpu wheel and CUDA user-space libs."
            )

        # Favor conservative arena growth to reduce VRAM spikes on large batches.
        providers: list[str | tuple[str, dict[str, str]]]
        if cuda_ready:
            providers = [
                (
                    "CUDAExecutionProvider",
                    {
                        "arena_extend_strategy": "kSameAsRequested",
                    },
                ),
                "CPUExecutionProvider",
            ]
        else:
            providers = ["CPUExecutionProvider"]

        session_options = rt.SessionOptions()
        session_options.enable_mem_pattern = False

        self.model = rt.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=providers,
        )

        if require_gpu and "CUDAExecutionProvider" not in self.model.get_providers():
            raise RuntimeError(
                "NSFW_TAGGER_REQUIRE_GPU is set, but ONNX Runtime did not "
                "activate CUDAExecutionProvider."
            )

        _, self.target_size, _, _ = self.model.get_inputs()[0].shape
        self.input_name = self.model.get_inputs()[0].name
        self.output_name = self.model.get_outputs()[0].name

    @staticmethod
    def _can_use_cuda_provider() -> bool:
        """Return True if CUDA provider is available and runtime deps are loadable.

        This avoids noisy ORT startup errors on machines where onnxruntime-gpu is
        installed but CUDA user-space libs (e.g., libcublasLt.so.12) are missing.
        """
        if os.getenv("NSFW_TAGGER_FORCE_CPU", "").strip().lower() in {"1", "true", "yes", "on"}:
            return False

        if "CUDAExecutionProvider" not in rt.get_available_providers():
            return False

        # Linux preflight: ORT will log an error if this shared object exists but
        # its transitive CUDA deps are missing. Probe it directly and fall back cleanly.
        cuda_provider_so = (
            Path(rt.__file__).resolve().parent / "capi" / "libonnxruntime_providers_cuda.so"
        )
        if cuda_provider_so.exists():
            mode = os.RTLD_NOW if hasattr(os, "RTLD_NOW") else None
            try:
                if mode is None:
                    ctypes.CDLL(str(cuda_provider_so))
                else:
                    ctypes.CDLL(str(cuda_provider_so), mode=mode)
            except OSError:
                return False
        elif find_library("cublasLt") is None:
            return False

        return True

    def _prepare_image(self, image_path: str | Path) -> np.ndarray:
        """Load and preprocess image for the tagger. Returns (1, H, W, 3) array."""
        image = Image.open(image_path).convert("RGBA")

        # Alpha composite onto white background
        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")

        # Pad to square
        w, h = image.size
        max_dim = max(w, h)
        pad_left = (max_dim - w) // 2
        pad_top = (max_dim - h) // 2
        padded = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        padded.paste(image, (pad_left, pad_top))

        # Resize
        if max_dim != self.target_size:
            padded = padded.resize((self.target_size, self.target_size), Image.BICUBIC)

        # Convert to float32 BGR
        arr = np.asarray(padded, dtype=np.float32)
        arr = arr[:, :, ::-1]  # RGB -> BGR
        return np.expand_dims(arr, axis=0)

    def classify(self, image_path: str | Path) -> ClassificationResult:
        """Classify a single image."""
        image_input = self._prepare_image(image_path)
        preds = self.model.run([self.output_name], {self.input_name: image_input})[0]
        return self._parse_predictions(preds[0])

    def classify_batch(self, image_paths: list[str | Path]) -> list[ClassificationResult]:
        """Classify a batch of images in one inference call."""
        from concurrent.futures import ThreadPoolExecutor

        workers = min(4, len(image_paths)) or 1
        with ThreadPoolExecutor(max_workers=workers) as pool:
            arrays = list(pool.map(lambda p: self._prepare_image(p)[0], image_paths))
        batch_input = np.stack(arrays, axis=0)
        preds = self.model.run([self.output_name], {self.input_name: batch_input})[0]
        return [self._parse_predictions(preds[i]) for i in range(len(image_paths))]

    def _parse_predictions(self, pred: np.ndarray) -> ClassificationResult:
        """Parse raw model output into a ClassificationResult."""
        labels = list(zip(self.tag_names, pred.astype(float), strict=True))

        # Rating scores
        rating_scores = {labels[i][0]: labels[i][1] for i in self.rating_indexes}
        top_rating = max(rating_scores, key=rating_scores.get)

        # Collect all general tags above threshold
        tags = {}
        for i in self.general_indexes:
            tag_name, score = labels[i]
            if score < TAG_THRESHOLD:
                continue
            tags[tag_name] = score

        content_rating = RATING_MAP.get(top_rating, "SAFE")

        return ClassificationResult(
            rating=top_rating,
            rating_scores={k: round(v, 4) for k, v in rating_scores.items()},
            content_rating=content_rating,
            tags={k: round(v, 4) for k, v in sorted(tags.items(), key=lambda x: -x[1])},
        )
