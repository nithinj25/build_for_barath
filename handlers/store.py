"""Where the API's state lives.

The serve bundle is read-only (built by linkage.bundle). Feedback and audit
records are append-only. Locally: a directory and JSON-lines files. On AWS:
the bundle in S3, feedback and audit in DynamoDB (spec §3.2 keys). boto3 is
imported only on the AWS path, and only in handlers/ (CLAUDE.md rule 2).
"""
from __future__ import annotations

import gzip
import json
import tempfile
from pathlib import Path

import numpy as np

BUNDLE_FILES = ("weights.json", "index.json", "codes.npz", "cases.json.gz", "meta.json")
OPTIONAL_FILES = ("truth_groups.json", "leads.json.gz", "series.json.gz", "checks.json.gz", "extras.npz", "phrases.json")


def load_bundle(uri: str) -> dict:
    root = _materialise(uri)
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    codes: dict = {}
    with np.load(root / "codes.npz") as arrays:
        for key in arrays.files:
            pool, field = key.split("|", 1)
            codes.setdefault(pool, {})[field] = arrays[key].astype(np.int64)
    with gzip.open(root / "cases.json.gz", "rt", encoding="utf-8") as fh:
        cases = json.load(fh)
    truth_path = root / "truth_groups.json"
    lists = {}
    for name in ("leads", "series", "checks"):
        path = root / f"{name}.json.gz"
        lists[name] = []
        if path.exists():
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                lists[name] = json.load(fh)
    hub_r = None
    if (root / "extras.npz").exists():
        with np.load(root / "extras.npz") as extras:
            hub_r = extras["hub_r"].astype(np.float64)
    phrases = json.loads((root / "phrases.json").read_text(encoding="utf-8")) if (root / "phrases.json").exists() else None
    return {
        **lists, "hub_r": hub_r, "phrases": phrases,
        "weights": json.loads((root / "weights.json").read_text(encoding="utf-8")),
        "case_ids": index["case_ids"], "crime_types": index["crime_types"],
        "codes": codes, "cases": cases,
        "meta": json.loads((root / "meta.json").read_text(encoding="utf-8")),
        "truth_groups": json.loads(truth_path.read_text(encoding="utf-8")) if truth_path.exists() else None,
    }


def _materialise(uri: str) -> Path:
    if not uri.startswith("s3://"):
        return Path(uri)
    import boto3
    from botocore.exceptions import ClientError

    bucket, _, prefix = uri[len("s3://"):].partition("/")
    dest = Path(tempfile.gettempdir()) / "bundle"
    dest.mkdir(exist_ok=True)
    s3 = boto3.client("s3")
    for name in (*BUNDLE_FILES, *OPTIONAL_FILES):
        key = f"{prefix.rstrip('/')}/{name}" if prefix else name
        try:
            s3.download_file(bucket, key, str(dest / name))
        except ClientError:
            if name in OPTIONAL_FILES:
                continue
            raise
    return dest


class LocalStore:
    """JSON-lines files; for development and tests."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _append(self, name: str, item: dict) -> None:
        with (self.root / f"{name}.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(item) + "\n")

    def _read(self, name: str) -> list[dict]:
        path = self.root / f"{name}.jsonl"
        return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l] if path.exists() else []

    def record_feedback(self, item: dict) -> None:
        self._append("feedback", item)

    def record_audit(self, item: dict) -> None:
        self._append("audit", item)

    def feedback_for(self, pair_id: str) -> list[dict]:
        return [f for f in self._read("feedback") if f["pair_id"] == pair_id]


class DynamoStore:
    """feedback: PK pair_id, SK actor#timestamp. audit: PK actor_id, SK timestamp."""

    def __init__(self, feedback_table: str, audit_table: str):
        import boto3

        dynamodb = boto3.resource("dynamodb")
        self.feedback = dynamodb.Table(feedback_table)
        self.audit = dynamodb.Table(audit_table)

    def record_feedback(self, item: dict) -> None:
        self.feedback.put_item(Item={**item, "sk": f"{item['actor_id']}#{item['timestamp']}"})

    def record_audit(self, item: dict) -> None:
        self.audit.put_item(Item=item)

    def feedback_for(self, pair_id: str) -> list[dict]:
        from boto3.dynamodb.conditions import Key

        return self.feedback.query(KeyConditionExpression=Key("pair_id").eq(pair_id)).get("Items", [])


def open_store(uri: str):
    """'local:<dir>' or 'dynamodb:<feedback table>,<audit table>'."""
    kind, _, rest = uri.partition(":")
    if kind == "local":
        return LocalStore(rest)
    if kind == "dynamodb":
        feedback_table, audit_table = rest.split(",")
        return DynamoStore(feedback_table, audit_table)
    raise ValueError(f"unknown store {uri!r}")
