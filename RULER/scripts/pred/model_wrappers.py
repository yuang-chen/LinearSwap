# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import requests
import torch
from typing import Dict, List, Optional


class HuggingFaceModel:
    def __init__(self, name_or_path: str, **generation_kwargs) -> None:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)

        if 'Yarn-Llama' in name_or_path:
            model_kwargs = None
        else:
            model_kwargs = {"attn_implementation": "flash_attention_2"}
        
        try:
            self.pipeline = pipeline(
                "text-generation",
                model=name_or_path,
                tokenizer=self.tokenizer,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                model_kwargs=model_kwargs,
            )
        except:
            self.pipeline = None
            self.model = AutoModelForCausalLM.from_pretrained(name_or_path, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16,)
            
        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop('stop')

        if self.tokenizer.pad_token is None:
            # add pad token to allow batching (known issue for llama2)
            self.tokenizer.padding_side = 'left'
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id


    def __call__(self, prompt: str, **kwargs) -> dict:
        return self.process_batch([prompt], **kwargs)[0]

    def process_batch(self, prompts: List[str], **kwargs) -> List[dict]:
        if self.pipeline is None:
            if self.tokenizer.chat_template is not None:
                prompts = [
                    self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": p}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for p in prompts
                ]
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
            generated_ids = self.model.generate(
                **inputs,
                **self.generation_kwargs
            )
            generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        else:
            output = self.pipeline(text_inputs=prompts, **self.generation_kwargs, )
            assert len(output) == len(prompts)
            # output in the form of a list of list of dictionaries
            # outer list len = batch size
            # inner list len = 1
            generated_texts = [llm_result[0]["generated_text"] for llm_result in output]

        results = []

        for text, prompt in zip(generated_texts, prompts):
            # remove the input form the generated text
            # This is a workaround for the llama3 tokenizer not being able to reproduce the same prompt after tokenization
            # see Issue https://github.com/NVIDIA/RULER/issues/54 for explaination
            if self.pipeline is None:
                tokenized_prompt = self.tokenizer(prompt, return_tensors="pt", padding=True)
                prompt = self.tokenizer.decode(tokenized_prompt.input_ids[0], skip_special_tokens=True)
            if text.startswith(prompt):
                text = text[len(prompt):]

            # Strip <think> blocks if present.
            if "<think>" in text and "</think>" in text:
                text = text.split("</think>")[-1]

            if self.stop is not None:
                for s in self.stop:
                    text = text.split(s)[0]

            results.append({'text': [text]})

        return results


class MambaModel:
    def __init__(self, name_or_path: str, **generation_kwargs) -> None:
        from transformers import AutoTokenizer
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        self.device = "cuda"
        self.model = MambaLMHeadModel.from_pretrained(name_or_path, device=self.device, dtype=torch.bfloat16)
        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop('stop')
        self.max_genlen = self.generation_kwargs.pop('max_new_tokens')
        self.minp = 0.0

    def __call__(self, prompt: str, **kwargs) -> Dict[str, List[str]]:
        # tokenize
        tokens = self.tokenizer(prompt, return_tensors="pt")
        input_ids = tokens.input_ids.to(self.device)
        max_length = input_ids.shape[1] + self.max_genlen

        # generate
        out = self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            cg=True,
            return_dict_in_generate=True,
            output_scores=True,
            enable_timing=False,
            **self.generation_kwargs,
        )
        assert len(out.sequences) == 1
        # detok
        return {'text': [self.tokenizer.decode(out.sequences[0][input_ids.shape[1]:])]}

    def process_batch(self, prompts: List[str], **kwargs) -> List[dict]:
        # FIXME: naive implementation
        return [self.__call__(prompt, **kwargs) for prompt in prompts]


class LinearSwapModelWrapper:
    """RULER wrapper for kernel-swapped hybrid models built with ``linswap``.

    ``name_or_path`` is a directory whose ``config.json`` records:
        linear_kernel   registry name of the kernel (e.g. "kda", "gdn2", "gdn", "deltanet")
        base_model_dir  (optional) HF backbone checkpoint used for the function-preserving init
        ckpt_dir        (optional) directory holding a native ``model.pt``; defaults to ``name_or_path``
    SFT checkpoints written by ``scripts/sft.py`` already carry these fields;
    ``scripts/register_ruler_model.py`` creates such a directory under
    ``outputs/ruler_models/<name>`` for a base (no-SFT) swap or any checkpoint.
    """

    def __init__(self, name_or_path: str, **generation_kwargs) -> None:
        import json
        import os
        import sys
        from pathlib import Path

        from transformers import AutoTokenizer

        try:
            from linswap import DEFAULT_BASE_MODEL_DIR, build_model
        except ImportError:  # not pip-installed: fall back to the checkout
            sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
            from linswap import DEFAULT_BASE_MODEL_DIR, build_model

        model_dir = Path(name_or_path)
        cfg = {}
        if (model_dir / "config.json").exists():
            with open(model_dir / "config.json") as f:
                cfg = json.load(f)
        kernel = cfg.get("linear_kernel") or os.environ.get("LINSWAP_KERNEL")
        if kernel is None:
            raise ValueError(f"{model_dir}/config.json has no 'linear_kernel' and LINSWAP_KERNEL is unset")
        # LINSWAP_BASE_MODEL_DIR (set by `linswap evaluate`) overrides the recorded path, which may be another machine's
        base_model_dir = Path(os.environ.get("LINSWAP_BASE_MODEL_DIR") or cfg.get("base_model_dir", DEFAULT_BASE_MODEL_DIR))
        if not base_model_dir.is_absolute():  # checkpoint configs record repo-relative paths; RULER runs from RULER/scripts
            base_model_dir = Path(__file__).resolve().parents[3] / base_model_dir
        ckpt_dir = Path(cfg["ckpt_dir"]) if cfg.get("ckpt_dir") else model_dir
        if not (ckpt_dir / "model.pt").exists():
            ckpt_dir = None
        print(f"[LinearSwapModelWrapper] kernel={kernel} base={base_model_dir} ckpt={ckpt_dir}")

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = torch.device("cuda")
        # The HF wrapper's generate handles left-padded batches (pads masked out of attention and of the linear
        # layers' state), per-row EOS and the cache.
        from linswap.hf import LinearSwapForCausalLM
        self.model = LinearSwapForCausalLM.from_swap(kernel, base_model_dir=base_model_dir, ckpt_dir=ckpt_dir,
                                                     device=self.device).eval()
        # Stop on end-of-turn (chat template) or end-of-text (base template).
        self.stop_ids = sorted({self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|endoftext|>")})
        # Rows per generate call: at most the batch call_api hands over, and at most LINSWAP_MAX_BATCH_TOKENS
        # padded tokens (prompt + generation) per call, so 128K prompts run one at a time at the default budget.
        # swa has no padding support (its sinks are absolute positions): one row at a time.
        self.max_batch_tokens = int(os.environ.get("LINSWAP_MAX_BATCH_TOKENS", 131072))
        self.max_rows = 1 if "swa" in kernel else None

        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop("stop")
        self.max_new_tokens = self.generation_kwargs.pop("max_new_tokens")
        self.use_chat_template = self.generation_kwargs.pop("use_chat_template", True) and \
            not os.environ.get("LINSWAP_NO_CHAT_TEMPLATE")   # base-prompt mode (RADLADS / KL-guided protocol)
        self.use_cache = self.generation_kwargs.pop("use_cache", True)

    def __call__(self, prompt: str, **kwargs) -> dict:
        return self.process_batch([prompt], **kwargs)[0]

    supports_answer_prefix = True  # call_api passes RULER's per-sample answer prefix separately

    def _render(self, prompt: str, answer_prefix: str) -> str:
        if self.tokenizer.chat_template is not None and self.use_chat_template:
            templated = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
                enable_thinking=False,  # thinking models (Qwen3.8) would spend the token budget on reasoning,
            )
            return templated + (answer_prefix or '').lstrip(' ')  # answer prefix opens the assistant turn
        return prompt + (answer_prefix or '')

    @torch.no_grad()
    def _generate(self, ids_list: List[List[int]]) -> List[str]:
        """Greedy decoding of one left-padded batch; returns the generated text per row."""
        enc = self.tokenizer.pad({"input_ids": ids_list}, return_tensors="pt").to(self.device)
        out = self.model.generate(
            input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
            max_new_tokens=self.max_new_tokens, do_sample=False,
            eos_token_id=self.stop_ids, pad_token_id=self.tokenizer.pad_token_id, use_cache=self.use_cache,
        )
        width = enc["input_ids"].shape[1]
        return [self.tokenizer.decode(row[width:], skip_special_tokens=True) for row in out]

    def process_batch(self, prompts: List[str], answer_prefixes=None, **kwargs) -> List[dict]:
        texts = [self._render(p, a) for p, a in zip(prompts, answer_prefixes or [''] * len(prompts))]
        ids = [self.tokenizer(t, padding=False)["input_ids"] for t in texts]
        # Longest first, so rows of similar length share a batch (little padding).
        order = sorted(range(len(ids)), key=lambda i: -len(ids[i]))
        texts_out = [None] * len(ids)
        start = 0
        while start < len(order):
            width = len(ids[order[start]]) + self.max_new_tokens
            n = max(1, min(self.max_rows or len(order), self.max_batch_tokens // width))
            chunk = order[start:start + n]
            try:
                generated = self._generate([ids[i] for i in chunk])
            except torch.OutOfMemoryError:
                if len(chunk) == 1:
                    raise
                generated = None
            if generated is None:
                # Halve the rows and keep the lower budget for the rest of the run.  Freed only here: the
                # exception's traceback pins the failed batch's tensors until the except block ends.
                torch.cuda.empty_cache()
                self.max_batch_tokens = width * (len(chunk) // 2)
                print(f"[LinearSwapModelWrapper] CUDA OOM at {len(chunk)} x {width} tokens; "
                      f"max_batch_tokens -> {self.max_batch_tokens}", flush=True)
                continue
            start += len(chunk)
            for i, text in zip(chunk, generated):
                texts_out[i] = text
        results = []
        for text in texts_out:
            # Strip <think> blocks and keep only the final answer.
            if "<think>" in text and "</think>" in text:
                text = text.split("</think>")[-1]
            if self.stop is not None:
                for s in self.stop:
                    text = text.split(s)[0]
            results.append({"text": [text]})
        return results
