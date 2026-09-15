# Runbook: LinearSwap experiments on an 8-GPU node (setup done; sections 1–5 are the jobs)

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
snapshot_download("Qwen/Qwen3.8-27B", local_dir="models/Qwen3.8-27B")     # 52 GB, only for the 27B jobs
PY
# RULER assets (SQuAD / HotpotQA; the word list and essays are in the repo)
(cd RULER/scripts/data/synthetic/json && bash download_qa_dataset.sh)
# SFT mixture and the FineWeb-Edu shard are prepared automatically by the first run that needs them
# (data/sft/len262144 ≈ 3.6 GB, data/text/fineweb-edu-1shard ≈ 2.9 GB; ~15 min each).
python tests/test_kernels.py && python tests/test_hf.py                  # sanity (≈ 5 min)
```

Mamba-2 (`mamba2`, `mamba2_beta`) *training* on Hopper needs Triton < 3.4 (the simple-GLA op
has no TileLang backend); make a second env for those jobs only:

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
T=niah_multikey_2,niah_multikey_3,niah_multiquery,vt,cwe,fwe,qa_1,qa_2   # hard RULER tasks
L=4096,16384,65536,131072
P=.venv/bin/python
```

## 1. Can a newer kernel beat GDN?  Equal-budget continued training, 1.5B tokens  (GPUs 0–3, ~15 h)

Identical recipe for `gdn`, `gdn2`, `kda`, `rwkv7`: exact swap → CE on FineWeb-Edu, 1.2B tokens at
8K then 300M at 16K → evaluate the checkpoint directly (base mode) and after the standard SFT.
Two shards of FineWeb-Edu (`--text_shards 2`) give ~1.5B tokens.

```bash
for i in 0 1 2 3; do k=$(echo gdn gdn2 kda rwkv7 | cut -d' ' -f$((i+1)));
nohup bash -c "export CUDA_VISIBLE_DEVICES=$i; \
 $P -m linswap distill --kernel $k --text_data fineweb-edu --text_shards 2 --text_mix 0.1 --stages ce \
    --max_length 2048 --batch_size 16 --grad_accum_steps 2 --ce_schedule 8192:1.2e9,16384:3e8 --ce_lr 1e-5 \
    --eval_batches 10 --eval_every 200 --save_every 2000 --output_dir outputs/cpt1p5b/$k/cpt > outputs/cpt1p5b_${k}_cpt.log 2>&1 \
 && C=\$(ls -d outputs/cpt1p5b/$k/cpt/checkpoint-* | sort -t- -k2 -n | tail -1) \
 && $P -m linswap posttrain --kernel $k --modes full --init_ckpt \$C --output_dir outputs/cpt1p5b/$k > outputs/cpt1p5b_${k}_sft.log 2>&1 \
 && $P -m linswap evaluate --models $k-cpt=\$C $k-cpt-sft-50=outputs/cpt1p5b/$k/sft_full/checkpoint-50 \
    --tasks $T --lengths $L --samples 50 --val_batches 0 --nll pg19,wikitext --name cpt1p5b-$k > outputs/eval/cpt1p5b-$k.log 2>&1; \
 echo DONE >> outputs/eval/cpt1p5b-$k.log" >/dev/null 2>&1 &
done
```
Read: `python tools/hard_tables.py outputs/eval/cpt1p5b-*.log` (or the `summary.md` in each
`outputs/eval/cpt1p5b-<k>/`).  If the four still tie within noise, that is the result.

## 2. Seeds for the headline comparison  (GPUs 4–5, ~3 h)

The review asked for three seeds; each full-SFT run is 10 minutes, the 4-length sweep ~100 min.

```bash
for s in 1 2; do gpu=$((3+s));
nohup bash -c "export CUDA_VISIBLE_DEVICES=$gpu; for k in gdn gdn2 kda rwkv7; do \
   $P -m linswap posttrain --kernel \$k --modes full --seed $s --output_dir outputs/seeds/s$s/\$k > outputs/seeds_s${s}_\$k.log 2>&1; done; \
 $P -m linswap evaluate --models gdn-s$s=outputs/seeds/s$s/gdn/sft_full/checkpoint-50 gdn2-s$s=outputs/seeds/s$s/gdn2/sft_full/checkpoint-50 \
   kda-s$s=outputs/seeds/s$s/kda/sft_full/checkpoint-50 rwkv7-s$s=outputs/seeds/s$s/rwkv7/sft_full/checkpoint-50 \
   --tasks $T --lengths $L --samples 50 --val_batches 0 --name seeds-s$s > outputs/eval/seeds-s$s.log 2>&1; echo DONE >> outputs/eval/seeds-s$s.log" >/dev/null 2>&1 &
done
```
(seed 42 is the existing run; report mean ± std over the three.)

## 3. Literature recipe for the remaining inexact kernels  (GPUs 6–7, ~7 h each)

`gla` (per-channel decay, no erase) and `mamba1` (needs `mamba_ssm`) with the same four-stage
recipe used for `mamba2` / `deltanet`; both train in `.venv` on any GPU.

```bash
for pair in "6 gla" "7 mamba1"; do set -- $pair; gpu=$1; k=$2;
nohup bash -c "export CUDA_VISIBLE_DEVICES=$gpu; \
 $P -m linswap distill --kernel $k --text_data fineweb-edu --text_mix 0.1 --stages layer,hidden,kl,ce --max_length 2048 --batch_size 16 \
    --grad_accum_steps 2 --layer_tokens 50e6 --hidden_tokens 50e6 --kl_tokens 300e6 --ce_length 16384 --ce_tokens 100e6 \
    --eval_batches 10 --eval_every 100 --save_every 500 --output_dir outputs/lit/$k/distill > outputs/lit/${k}_distill.log 2>&1 \
 && C=\$(ls -d outputs/lit/$k/distill/checkpoint-* | sort -t- -k2 -n | tail -1) \
 && $P -m linswap posttrain --kernel $k --modes full --init_ckpt \$C --output_dir outputs/lit/$k > outputs/lit/${k}_sft.log 2>&1 \
 && $P -m linswap evaluate --models $k-lit-distill=\$C $k-lit-distill-sft-50=outputs/lit/$k/sft_full/checkpoint-50 \
    --tasks $T --lengths $L --samples 50 --val_batches 0 --nll pg19,wikitext --name lit-$k > outputs/eval/lit-$k.log 2>&1; \
 echo DONE >> outputs/eval/lit-$k.log" >/dev/null 2>&1 &
done
```

## 4. After (2) frees GPUs 4–5: 27B hard tasks at 128K with 50 samples  (needs ≥ 80 GB per GPU, ~8 h)

The 27B gate-only checkpoints live on the current server (`outputs/qwen38-27b/{kda,rwkv7}/sft_gate_only/checkpoint-100`,
~110 MB each of new parameters + the full state dict; copy the two `checkpoint-100` directories over), or re-train
them there (~40 min each):

```bash
# re-train (optional):  CUDA_VISIBLE_DEVICES=4 $P -m linswap posttrain --kernel kda --base_model_dir models/Qwen3.8-27B \
#     --modes gate_only --gate_steps 100 --max_length 32768 --eval_every 50 --save_every 50 --eval_batches 5 --save_optimizer 0 \
#     --output_dir outputs/qwen38-27b/kda
nohup bash -c "export CUDA_VISIBLE_DEVICES=4; $P -m linswap evaluate --base_model_dir models/Qwen3.8-27B \
  --models gdn kda-gate-100=outputs/qwen38-27b/kda/sft_gate_only/checkpoint-100 rwkv7-gate-100=outputs/qwen38-27b/rwkv7/sft_gate_only/checkpoint-100 \
  --tasks $T --lengths 16384,65536,131072 --samples 50 --val_batches 0 --name qwen38-27b-hard > outputs/eval/qwen38-27b-hard.log 2>&1; \
  echo DONE >> outputs/eval/qwen38-27b-hard.log" >/dev/null 2>&1 &
```

## 5. Short-context suites for every checkpoint produced above  (any free GPU, ~30 min per model)

```bash
CUDA_VISIBLE_DEVICES=5 $P -m linswap lmeval --models <label=path ...> --name csr-node          # PIQA … LAMBADA (GDN paper Table 3)
CUDA_VISIBLE_DEVICES=5 $P -m linswap lmeval --batch_size 1 --limit 500 --tasks swde,fda,squad_completion --models <...> --name recall-node
CUDA_VISIBLE_DEVICES=5 $P -m linswap mqar --models <...> --pairs 64,256,1024,4096 --name mqar-node
```

## Notes

* Everything writes under `outputs/`; `outputs/eval/<name>/summary.md` has the per-task table, the
  `*.log` files have the per-task timings.  Nothing needs to be copied back except `outputs/eval`
  and the `train_log.jsonl` files (checkpoints are 1.5 GB each; keep only the ones you want to reuse).
* The hard-task RULER numbers are ±7 points at 50 samples; do not read differences below that.
* If a RULER task fails, `evaluate` records it as FAILED and continues; the log names the RULER log to inspect.
