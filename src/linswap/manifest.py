"""Run manifests: what a training run was made of, written next to its checkpoints.

``linswap distill`` writes ``manifest.json`` with the code revision, the versions of the packages that
decide the numerics, the identity of the base model, the data mix and held-out set, and the GPU it ran on
with whatever else was running there (a GPU-hour measured on a shared GPU is not one).  ``linswap eval``
stores the revision and package versions with every suite.  Nothing here is read back by training; it is
the record that lets a result be traced to its inputs.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from importlib import metadata
from pathlib import Path

MANIFEST_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGES = ("torch", "triton", "transformers", "flash-linear-attention", "datasets", "tokenizers", "tilelang",
            "safetensors", "numpy", "lm_eval", "lmms_eval", "mamba_ssm")
TOKENIZER_FILES = ("tokenizer.json", "vocab.json", "tokenizer.model", "spiece.model")


def file_sha256(path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def tokenizer_fingerprint(model_dir) -> str | None:
    """sha256 of the file that defines the vocabulary and its segmentation (``tokenizer.json`` for every
    fast tokenizer), or None when ``model_dir`` holds none of the known files."""
    for name in TOKENIZER_FILES:
        path = Path(model_dir) / name
        if path.exists():
            return file_sha256(path)
    return None


def git_revision(repo=REPO_ROOT) -> dict:
    def git(*cmd):
        return subprocess.run(["git", "-C", str(repo), *cmd], capture_output=True, text=True, timeout=30)

    try:
        head = git("rev-parse", "HEAD")
        if head.returncode:
            return {}
        return {"git_commit": head.stdout.strip(), "git_dirty": bool(git("status", "--porcelain", "-uno").stdout.strip())}
    except (OSError, subprocess.SubprocessError):
        return {}


def package_versions(names=PACKAGES) -> dict:
    """Installed version of each package; a package installed from git also gets its commit, which the
    version string alone does not identify (``flash-linear-attention`` 0.6.0 is many revisions)."""
    out = {}
    for name in names:
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        out[name] = dist.version
        try:
            commit = json.loads(dist.read_text("direct_url.json") or "{}").get("vcs_info", {}).get("commit_id")
        except ValueError:
            commit = None
        if commit:
            out[name] += f"+git.{commit[:12]}"
    return out


def model_identity(model_dir) -> dict:
    """The base checkpoint: hashes of its config and tokenizer, and for each weight file its size plus,
    where ``hf download`` left its metadata, the Hub commit and the file's etag (the sha256 of an LFS file)
    — hashing the weights themselves would cost minutes on the larger backbones."""
    model_dir = Path(model_dir)
    out = {"dir": str(model_dir), "tokenizer_sha256": tokenizer_fingerprint(model_dir)}
    if (model_dir / "config.json").exists():
        out["config_sha256"] = file_sha256(model_dir / "config.json")
    weights = {}
    for path in sorted(model_dir.glob("*.safetensors")):
        entry = {"bytes": path.stat().st_size}
        meta = model_dir / ".cache" / "huggingface" / "download" / f"{path.name}.metadata"
        if meta.exists():
            lines = meta.read_text().split("\n")
            if len(lines) >= 2:
                out["hub_commit"], entry["etag"] = lines[0].strip(), lines[1].strip()
        weights[path.name] = entry
    out["weights"] = weights
    return out


def gpu_state(device=0) -> dict:
    """The GPU a run is on — name, index, memory in use, utilisation — and the other compute processes on it
    when the run starts.  Jobs of another container may not be listed, hence memory and utilisation."""
    import os

    import torch

    def smi(query):
        try:
            done = subprocess.run(["nvidia-smi", f"--query-{query}", "--format=csv,noheader,nounits"], capture_output=True,
                                  text=True, timeout=30)
            return [line.split(", ") for line in done.stdout.splitlines()]
        except (OSError, subprocess.SubprocessError):
            return []

    props = torch.cuda.get_device_properties(device)
    uuid = str(props.uuid)                                      # nvidia-smi prints it as GPU-<uuid>
    out = {"name": props.name, "uuid": uuid, "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    for row in smi("gpu=index,uuid,memory.used,utilization.gpu"):
        if len(row) == 4 and row[1].endswith(uuid):
            out.update(index=int(row[0]), memory_used_mib=row[2], utilization=row[3])
    out["other_processes"] = [dict(zip(("pid", "memory_mib", "name"), row[1:]))
                              for row in smi("compute-apps=gpu_uuid,pid,used_memory,process_name")
                              if len(row) == 4 and row[0].endswith(uuid) and row[1] != str(os.getpid())]
    return out


def write_manifest(path, **sections) -> dict:
    manifest = {"manifest_version": MANIFEST_VERSION, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "code": git_revision(), "packages": package_versions(), **sections}
    with open(path, "w") as f:
        json.dump(manifest, f, indent=1, default=str)
    return manifest
