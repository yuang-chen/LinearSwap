# Runbook: LinearSwap on an 8-GPU node

Everything below is one process per GPU (the 0.8B backbone never needs more than one GPU;
the trainer is single-GPU).  `CUDA_VISIBLE_DEVICES=<i>` selects the GPU; jobs are launched
with `nohup ... &` and write a `*_DONE` marker into their log when they finish.  Times are for
an L20X (≈ H100-PCIe class); an H100-SXM is ~1.5–2× faster.

## 0. One-time setup (≈ 30 min, network needed)

```bash
git clone https://github.com/yuang-chen/LinearSwap && cd LinearSwap     # needs the latest push
uv venv .venv --python 3.11 && source .venv/bin/activate
uv pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install "flash-linear-attention @ git+https://github.com/fla-org/flash-linear-attention@8e84ed4"
uv pip install -e ".[eval]" pytest
# Hopper GPUs (H100/H200, compute capability 9.x): FLA refuses the Triton backward of the gated
# chunk kernels under Triton 3.4–3.7 -> TileLang supplies it for gdn / gdn2 / kda:
uv pip install -e ".[hopper]"
# optional: fast short conv + mamba1/mamba3 kernels (build without dependency resolution!)
CUDA_HOME=/usr/local/cuda MAX_JOBS=32 uv pip install --no-deps --no-build-isolation \
    --no-binary causal-conv1d --no-binary mamba-ssm causal-conv1d mamba-ssm

# backbones
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3.5-0.8B", local_dir="models/Qwen3.5-0.8B")
PY
# RULER assets (SQuAD / HotpotQA; the word list and essays are in the repo)
(cd RULER/scripts/data/synthetic/json && bash download_qa_dataset.sh)
# The DCLM corpus is tokenised automatically by the first run that needs it
# (10 shards ≈ 1.1B tokens, ~20 min); start one job first so the others reuse it.
python tests/test_kernels.py && python tests/test_hf.py                  # sanity (≈ 5 min)
```

Mamba-2 *training* on Hopper needs Triton < 3.4 (the simple-GLA op has no TileLang backend);
make a second env for that job only:

```bash
uv venv .venv-torch26 --python 3.11
uv pip install --python .venv-torch26/bin/python torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv-torch26/bin/python "flash-linear-attention @ git+https://github.com/fla-org/flash-linear-attention@8e84ed4"
uv pip install --python .venv-torch26/bin/python -e ".[eval]"
# run mamba2 training as:  PYTHONPATH=src .venv-torch26/bin/python -m linswap distill ...
```
On Ampere (A100) none of this is needed: everything trains in `.venv`.

Common shell variables used below:

```bash
P=.venv/bin/python
```

## 1. One kernel per GPU: the standard pipeline  (~6 h training + ~3 h evaluation each)

Each chain is verify → distill (three steps, 700M DCLM tokens) → RULER → lm-eval.  Put the
unswapped `gdn` control on one of the GPUs: it is what the swapped rows are compared against.

```bash
for pair in "0 gdn" "1 rwkv7" "2 mamba2" "3 kda" "4 gdn2" "5 deltanet" "6 gla" "7 mamba1"; do
  set -- $pair; gpu=$1; k=$2
  nohup bash -c "export CUDA_VISIBLE_DEVICES=$gpu; $P -m linswap run --kernel $k --name $k \
      > outputs/run_$k.log 2>&1; echo DONE >> outputs/run_$k.log" >/dev/null 2>&1 &
done
```

`run` writes `outputs/<kernel>/distill/checkpoint-*`, `outputs/eval/<kernel>/summary.{csv,md,json}`
and `outputs/eval/<kernel>-lmeval/lmeval.{csv,md,json}`.  Read them with

```bash
python tools/hard_tables.py outputs/run_*.log          # retrieval, one table per length
cat outputs/eval/*-lmeval/lmeval.md                    # short-context suite with relative scores
```

On Hopper, run the `mamba2` chain in the Triton-3.2 environment:
`PYTHONPATH=src .venv-torch26/bin/python -m linswap distill --kernel mamba2 ...`, then evaluate it
with the main environment.

## 2. Seeds  (any spare GPU, ~9 h each)

`--seed 1` / `--seed 2` on the kernels whose margin you want an error bar for; three seeds of the
control and of `rwkv7` is the cheapest useful replication.

## 3. Larger token budgets  (~15 h per kernel at 1.5B tokens)

```bash
$P -m linswap distill --kernel rwkv7 --text_shards 20 --kl_tokens 1.2e9 --ce_tokens 3e8 \
    --output_dir outputs/rwkv7-1p5b/distill
```

Everything else (evaluation commands, reading the tables) is unchanged.

## 4. Throughput

```bash
python tools/throughput.py --models gdn rwkv7=outputs/rwkv7/distill/checkpoint-16338 --lengths 8192,32768
```

## Notes

* Everything writes under `outputs/`; only `outputs/eval` and the `train_log.jsonl` files need to come
  back from the node (checkpoints are 1.5 GB each).
* RULER scores are ±7 points at 50 samples; do not read differences below that.
* If a RULER task fails, `evaluate` records it as FAILED and continues; the log names the RULER log.
