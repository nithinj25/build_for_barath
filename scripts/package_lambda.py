"""Stage the Lambda package: python scripts/package_lambda.py [--bundle data/serve]

    build/app/     handlers/, the pure-numpy linkage modules, bundle/, ui/
    build/layer/   numpy for Linux arm64, fetched as a manylinux wheel with pip

No Docker: pip downloads the Linux wheel directly (--platform), so building
on Windows produces a layer that runs on Lambda. Only the modules the API
imports are copied — no pandas, sklearn or scipy reach Lambda (CLAUDE.md
rule 3). The case bundle ships inside the package, so the free stack needs no
S3 bucket of its own.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP, LAYER = ROOT / "build/app", ROOT / "build/layer"
MODULES = ["handlers/__init__.py", "handlers/api.py", "handlers/store.py",
           "linkage/__init__.py", "linkage/schema.py", "linkage/features.py",
           "linkage/score.py", "linkage/serve.py", "linkage/rank.py", "linkage/checks.py", "ui/index.html"]
BUNDLE_FILES = ["weights.json", "index.json", "codes.npz", "cases.json.gz", "meta.json", "truth_groups.json", "leads.json.gz", "series.json.gz", "checks.json.gz", "extras.npz"]
NUMPY = "numpy==1.26.4"            # the version every local test and check ran against
LAMBDA_UNZIPPED_LIMIT_MB = 250


def size_mb(path: Path) -> float:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e6


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bundle", type=Path, default=ROOT / "data/serve")
    args = ap.parse_args()

    for d in (APP, LAYER):
        shutil.rmtree(d, ignore_errors=True)
    for rel in MODULES:
        (APP / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, APP / rel)
    (APP / "bundle").mkdir()
    for name in BUNDLE_FILES:
        if (args.bundle / name).exists():
            shutil.copy2(args.bundle / name, APP / "bundle" / name)
        elif name not in ("truth_groups.json", "leads.json.gz", "series.json.gz", "checks.json.gz", "extras.npz"):
            print(f"missing bundle file {name} — run python -m linkage.bundle first")
            return 1

    subprocess.run([sys.executable, "-m", "pip", "install", NUMPY, "--quiet", "--disable-pip-version-check",
                    "--platform", "manylinux2014_aarch64", "--only-binary=:all:", "--python-version", "3.12",
                    "--implementation", "cp", "--target", str(LAYER / "python")], check=True)
    for junk in ("bin", "numpy/tests", "numpy/*/tests"):         # tests are not needed at runtime
        for p in LAYER.glob(f"python/{junk}"):
            shutil.rmtree(p, ignore_errors=True)

    app_mb, layer_mb = size_mb(APP), size_mb(LAYER)
    print(f"build/app   {app_mb:6.1f} MB  ({len(MODULES)} source files + bundle)")
    print(f"build/layer {layer_mb:6.1f} MB  ({NUMPY}, linux arm64)")
    print(f"unzipped total {app_mb + layer_mb:.1f} MB of Lambda's {LAMBDA_UNZIPPED_LIMIT_MB} MB limit")
    if not smoke_test():
        return 1
    return 0 if app_mb + layer_mb < LAMBDA_UNZIPPED_LIMIT_MB else 1


SMOKE = """
import os, sys, tempfile
sys.path.insert(0, ".")
os.environ.update(BUNDLE_URI="bundle", STORE="local:" + tempfile.mkdtemp())
from handlers import api
for path in ("/api/meta", "/api/leads", "/api/series"):
    r = api.handler({"rawPath": path, "requestContext": {"http": {"method": "GET"}}})
    assert r["statusCode"] == 200, (path, r["statusCode"], r["body"][:200])
print("staged app answers /meta, /leads, /series")
"""


def smoke_test() -> bool:
    """Import and call the STAGED app from its own folder, so a module left out
    of MODULES fails here instead of as a 502 on Lambda."""
    r = subprocess.run([sys.executable, "-c", SMOKE], cwd=APP, capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip().splitlines()[-1])
    return r.returncode == 0


if __name__ == "__main__":
    sys.exit(main())
