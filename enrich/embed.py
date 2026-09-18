"""Narrative embeddings, computed once and stored.

    python -m enrich.embed --normalised data/final_sharing1_normalised.parquet \\
        --out data/final_sharing1_vectors

Retrieval uses narrative-embedding cosine ONLY; scoring uses structured MO
only (CLAUDE.md rule 1). Vectors are built in batch and saved, as spec §6's
BuildIndex step does, so querying a corpus case needs no embedding call.

Model: intfloat/multilingual-e5-base — 100+ languages in one space, standing
in for Cohere multilingual (spec §5), 768-dim, fast on a laptop GPU. e5 wants
a "passage: " prefix on documents. Vectors are L2-normalised, so cosine is a
dot product, and stored as float16 (44.5k × 768 ≈ 68 MB).

Caveat that travels with every retrieval number: this corpus's narratives
are templated from the same recorded MO values the scorer uses, so retrieval
here is easier than it will be on real FIRs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

MODEL = "intfloat/multilingual-e5-base"


def embed_texts(texts: list[str], model_name: str = MODEL, batch_size: int = 128) -> np.ndarray:
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    if device == "cuda":
        model.half()
    vectors = model.encode([f"passage: {t}" for t in texts], batch_size=batch_size,
                           normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
    return vectors.astype(np.float16)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m enrich.embed", description=__doc__.splitlines()[0])
    ap.add_argument("--normalised", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="directory for vectors.npy + case_ids.json")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    norm = pd.read_parquet(args.normalised, columns=["case_id", "narrative_text"])
    started = time.time()
    vectors = embed_texts(norm["narrative_text"].fillna("").tolist(), args.model)
    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "vectors.npy", vectors)
    (args.out / "case_ids.json").write_text(json.dumps(norm["case_id"].tolist()), encoding="utf-8")
    (args.out / "meta.json").write_text(json.dumps({
        "model": args.model, "dim": int(vectors.shape[1]), "count": int(len(vectors)),
        "dtype": "float16", "normalised": True, "prefix": "passage: ",
        "source": str(args.normalised), "seconds": round(time.time() - started, 1)}, indent=2), encoding="utf-8")
    print(f"embedded {len(vectors)} narratives → {args.out} ({vectors.shape[1]}-dim, {time.time() - started:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
