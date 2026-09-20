"""Model engine: loading, templating, states, prefix scoring, patch forwards, decoding."""

from __future__ import annotations

import gc
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

import corpus

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 512
# continuation scoring materializes [R, len, V] logits at the LM head (fp32 on
# the GPU).  Iteration-1 fixed CONT_SCORE_CHUNK=48 (~20GB VRAM target); the
# iteration-2 hardware (RTX 2000 Ada, 16GB) and Gemma-3-4b-it (V=262144)
# require an adaptive chunk.  Formula (plan step 0): chunk = max(8, min(48,
# int(8e9 / (maxlen * vocab * 4)))); halved further on OOM inside
# ModelHandle.score_prefixes.
CONT_SCORE_CHUNK = 48
# bytes of fp32 logits per TENSOR we allow on the GPU at once: logsumexp(lg)
# materializes a SECOND [R, maxlen, V] tensor, so peak = 2 * chunk * per_row +
# weights + hidden states.  2.2e9 keeps the pair <= ~4.4 GB on a 16 GB GPU.
_CONT_LOGITS_BUDGET = 2.2e9


def adaptive_cont_chunk(maxlen: int, vocab: int) -> int:
    """Chunk rows of [R, maxlen, V] fp32 logits to fit the GPU budget."""
    try:
        per_row = max(1, maxlen * max(1, vocab) * 4)
        return max(8, min(48, int(_CONT_LOGITS_BUDGET / per_row)))
    except Exception:
        return 24


@dataclass
class PromptEncoding:
    """Token-level geometry for one rendered prompt."""
    no_gen: list[int]      # ids without assistant header
    full: list[int]        # ids with assistant header
    decision_pos: int      # last user-turn token index (== len(no_gen)-1)
    first_gen_pos: int     # index of last assistant-header token (== len(full)-1)


class ModelHandle:
    def __init__(
        self,
        model: Any,
        tok: AutoTokenizer,
        family: str,
        model_id: str,
        cls: str,
        base_cfg: dict[str, Any] | None = None,
    ):
        self.model = model
        self.tok = tok
        self.family = family
        self.model_id = model_id
        self.cls = cls
        cfg = model.config

        def _cfg_int(*names: str) -> int:
            """Resolve an integer config field, falling back to the nested
            language-model 'text_config' (Gemma3ForConditionalGeneration hides
            num_hidden_layers/hidden_size/vocab_size there)."""
            for _n in names:
                v = getattr(cfg, _n, None)
                if v is not None:
                    return int(v)
            tc = getattr(cfg, "text_config", None)
            if tc is not None:
                for _n in names:
                    v = getattr(tc, _n, None)
                    if v is not None:
                        return int(v)
            return 0

        self.L: int = _cfg_int("num_hidden_layers", "num_layers")
        self.D: int = _cfg_int("hidden_size", "d_model")
        self.eos = tok.eos_token_id
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id
        self.pad = tok.pad_token_id
        # Module holding the per-layer decoder ("model.model.layers" for plain
        # causal LMs; Gemma3ForConditionalGeneration nests the text decoder at
        # model.model.language_model.layers instead).
        _lm_cand = getattr(model, "model", None)
        if _lm_cand is not None and hasattr(_lm_cand, "language_model") \
                and hasattr(getattr(_lm_cand, "language_model", None), "layers"):
            _lm_cand = _lm_cand.language_model
        self._layer_module = _lm_cand
        self.n_params = sum(p.numel() for p in model.parameters())
        self.base_cfg = base_cfg or {}
        logger.info(
            f"handle {model_id} family={family} cls={cls} L={self.L} D={self.D} "
            f"params={self.n_params/1e9:.2f}B"
        )
        # Precompute token ids of prefix candidates (per family vocab)
        self._cand_cache: dict[str, list[int]] = {}

    # ------------------------------------------------------------------ io
    def encode(self, text: str, with_gen: bool) -> list[int]:
        r = corpus.render(text, with_gen, self.family)
        return self.tok(r, add_special_tokens=False)["input_ids"]

    def encode_prompt(self, text: str) -> PromptEncoding:
        ng = self.encode(text, with_gen=False)
        fl = self.encode(text, with_gen=True)
        return PromptEncoding(
            no_gen=ng, full=fl,
            decision_pos=len(ng) - 1, first_gen_pos=len(fl) - 1,
        )

    def cand_tokens(self, cand: str) -> list[int]:
        if cand not in self._cand_cache:
            self._cand_cache[cand] = self.tok(cand, add_special_tokens=False)["input_ids"]
        return self._cand_cache[cand]

    def _text_layer_module(self, patch_layer: int) -> Any:
        """The decoder-layer module at index patch_layer, architecture-aware.

        Standard causal LMs expose model.model.layers; Gemma3ForConditional-
        Generation nests the text decoder at model.model.text_model.layers
        (and keeps the vision tower alongside).  Additive: standard models
        resolve through the identical path as iteration-1 code."""
        base = getattr(self.model, "model", None)
        if base is not None:
            if hasattr(base, "layers"):
                return base.layers[patch_layer]
            for attr in ("text_model", "language_model", "model"):
                sub = getattr(base, attr, None)
                if sub is not None and hasattr(sub, "layers"):
                    return sub.layers[patch_layer]
            raise AttributeError(f"cannot locate decoder layers for {self.model_id}")
        return self.model.layers[patch_layer]

    # ----------------------------------------------------------- forwards
    @torch.no_grad()
    def forward_states(self, encs: list[PromptEncoding], batch_size: int = 12) -> dict[str, Any]:
        """One forward per batch; returns post-layer hidden states at decision
        positions, first-gen logits, and the ids tensors."""
        n = len(encs)
        L, D = self.L, self.D
        H = np.zeros((L, n, D), dtype=np.float32)      # H[l] = after layer l (0-based l)
        first_logits: list[np.ndarray] = []
        cur_batch = max(1, int(batch_size))
        b0 = 0
        while b0 < n:
            batch = encs[b0 : b0 + cur_batch]
            maxlen = max(len(e.full) for e in batch)
            ids = torch.full((len(batch), maxlen), self.pad, dtype=torch.long)
            am = torch.zeros((len(batch), maxlen), dtype=torch.long)
            fgp = torch.tensor([e.first_gen_pos for e in batch], dtype=torch.long)
            for i, e in enumerate(batch):
                ids[i, : len(e.full)] = torch.tensor(e.full, dtype=torch.long)
                am[i, : len(e.full)] = 1
            ids, am, fgp = ids.to(DEVICE), am.to(DEVICE), fgp.to(DEVICE)
            try:
                out = self.model(
                    input_ids=ids, attention_mask=am,
                    output_hidden_states=True, use_cache=False,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if cur_batch > 1:
                    cur_batch = max(1, cur_batch // 2)
                    logger.warning(f"forward_states OOM: halved batch to {cur_batch} for {self.model_id}")
                    del ids, am, fgp
                    continue
                raise
            hs = out.hidden_states  # list len L+1, [B,T,D]
            lg = out.logits.float()  # [B,T,V]
            gl = lg[torch.arange(len(batch)), fgp]  # [B,V]
            for i, e in enumerate(batch):
                dp = e.decision_pos
                for l in range(L):
                    H[l, b0 + i] = hs[l + 1][i, dp].float().cpu().numpy()  # after layer l
                first_logits.append(gl[i].cpu().numpy())
            del out, hs, lg, gl
            b0 += len(batch)
        return {
            "H": H,
            "first_logits": np.stack(first_logits),
            "encs": encs,
            "n": n,
        }

    # ----------------------------------------------------- prefix scoring
    @torch.no_grad()
    def score_prefixes(
        self,
        encs: list[PromptEncoding],
        first_logits: np.ndarray,
        candidates: list[str],
        patch_rows: list[dict[str, Any]] | None = None,
        patch_layer: int | None = None,
    ) -> np.ndarray:
        """Teacher-forced log-prob of each candidate prefix per prompt.

        patch_rows: optional list of {'row': int, 'z': torch.Tensor (cuda fp16)}
        -- z is injected at encs[row].decision_pos before patch_layer runs.
        Returns logp [n_prompts, n_candidates] (natural log)."""
        n = len(encs)
        nc = len(candidates)
        logp = np.full((n, nc), -1e9, dtype=np.float64)

        c0_ids: list[list[int]] = [self.cand_tokens(c) for c in candidates]

        if patch_rows:
            # collect per-row z (applies to the batch of all rows)
            zmap = {pr["row"]: pr["z"] for pr in patch_rows}
        else:
            zmap = None

        def _make_hook(zz: list[torch.Tensor], dp: list[int], fired: dict):
            def hook_fn(module, args):
                x = args[0].clone()
                for k in range(x.shape[0]):
                    if k < len(zz):
                        x_k = x[k]
                        x_k[dp[k]] = zz[k]
                fired["n"] += 1
                return x
            return hook_fn

        # score first token of each candidate from first-gen logits
        logits0 = torch.from_numpy(first_logits).float().to(DEVICE)
        lse0 = torch.logsumexp(logits0, dim=-1, keepdim=True)
        p0 = (logits0 - lse0).cpu().numpy()  # [n, V]
        for j, ids in enumerate(c0_ids):
            if not ids:
                continue
            logp[:, j] = p0[:, ids[0]]

        # continuation scoring: input = full + c[:-1] for (prompt, candidate),
        # row-chunked to bound the [R, len, V] logits tensor on the GPU.
        rows: list[tuple[int, int]] = [
            (i, j) for i in range(n) for j, ids in enumerate(c0_ids) if len(ids) > 1
        ]
        cont_all = [encs[i].full + c0_ids[j][:-1] for (i, j) in rows]
        _maxlen = max(len(r) for r in cont_all) if cont_all else 1
        _mcfg = getattr(self.model, "config", None)
        _vocab = max(1, int(getattr(_mcfg, "vocab_size", 0) or 0) or 0)
        if not _vocab and getattr(_mcfg, "text_config", None) is not None:
            _vocab = int(getattr(_mcfg.text_config, "vocab_size", 0) or 0)
        _vocab = max(1, _vocab)
        _chunk = adaptive_cont_chunk(_maxlen, _vocab)
        if _chunk < CONT_SCORE_CHUNK:
            logger.warning(
                f"adaptive CONT_SCORE_CHUNK {CONT_SCORE_CHUNK} -> {_chunk} "
                f"(maxlen={_maxlen} vocab={_vocab}) for {self.model_id}"
            )
        del cont_all
        b0 = 0
        while b0 < len(rows):
            chunk = rows[b0 : b0 + _chunk]
            cont = [encs[i].full + c0_ids[j][:-1] for (i, j) in chunk]
            maxlen = max(len(r) for r in cont)
            ids = torch.full((len(chunk), maxlen), self.pad, dtype=torch.long)
            am = torch.zeros((len(chunk), maxlen), dtype=torch.long)
            for r, seq in enumerate(cont):
                ids[r, : len(seq)] = torch.tensor(seq, dtype=torch.long)
                am[r, : len(seq)] = 1
            ids, am = ids.to(DEVICE), am.to(DEVICE)

            hook = None
            fired = {"n": 0}
            if zmap is not None:
                assert patch_layer is not None
                zs = [zmap[i] for i, _ in chunk]
                dp = [encs[i].decision_pos for i, _ in chunk]
                hook = self._text_layer_module(patch_layer).register_forward_pre_hook(
                    _make_hook(zs, dp, fired)
                )
            try:
                try:
                    out = self.model(input_ids=ids, attention_mask=am, use_cache=False, output_hidden_states=False)
                    lg = out.logits.float()  # [R, maxlen, V]
                    seql = torch.logsumexp(lg, dim=-1)  # [R, maxlen] (second big tensor)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if _chunk > 8:
                        _chunk = max(8, _chunk // 2)
                        logger.warning(f"score_prefixes OOM: halved chunk to {_chunk} for {self.model_id}")
                        del ids, am
                        continue
                    raise
            finally:
                if hook is not None:
                    hook.remove()
            if zmap is not None:
                # the pre-forward hook fires once per forward pass (one batch here)
                assert fired["n"] >= 1, f"patch hook never fired"

            # score tokens of c[1:] at appended positions
            for r, (i, j) in enumerate(chunk):
                ids_c = c0_ids[j]
                base = len(encs[i].full)
                acc = 0.0
                for k, tok_id in enumerate(ids_c[1:]):
                    pos = base + k
                    acc += float((lg[r, pos, tok_id] - seql[r, pos]).item())
                logp[i, j] += acc
            del out, lg, seql, ids, am
            b0 += len(chunk)

        return logp

    # ------------------------------------------------------- d (contrast)
    @torch.no_grad()
    def refusal_intent(
        self,
        encs: list[PromptEncoding],
        first_logits: np.ndarray,
        patch_rows: list[dict[str, Any]] | None = None,
        patch_layer: int | None = None,
    ) -> np.ndarray:
        """d_i = max_R logP(prefix) - max_C logP(prefix) per prompt."""
        lpR = self.score_prefixes(encs, first_logits, corpus.REFUSAL_CANDIDATES, patch_rows, patch_layer)
        lpC = self.score_prefixes(encs, first_logits, corpus.COMPLIANCE_CANDIDATES, patch_rows, patch_layer)
        return np.max(lpR, axis=1) - np.max(lpC, axis=1)

    # ------------------------------------------------------ decode
    @torch.no_grad()
    def decode_batch(
        self,
        encs: list[PromptEncoding],
        max_new: int = 160,
        batch_size: int = 8,
    ) -> list[str]:
        """Greedy decoding; returns text after the assistant header."""
        out_texts: list[str] = [""] * len(encs)
        for b0 in range(0, len(encs), batch_size):
            batch = encs[b0 : b0 + batch_size]
            maxlen = max(len(e.full) for e in batch)
            ids = torch.full((len(batch), maxlen), self.pad, dtype=torch.long)
            am = torch.zeros((len(batch), maxlen), dtype=torch.long)
            for i, e in enumerate(batch):
                pad_n = maxlen - len(e.full)
                ids[i, pad_n:] = torch.tensor(e.full, dtype=torch.long)
                am[i, pad_n:] = 1
            ids, am = ids.to(DEVICE), am.to(DEVICE)
            gen = self.model.generate(
                input_ids=ids,
                attention_mask=am,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=self.pad,
                eos_token_id=self.eos,
                use_cache=True,
            )
            for i, e in enumerate(batch):
                pad_n = maxlen - len(e.full)
                tail = gen[i, pad_n + e.first_gen_pos :]
                text = self.tok.decode(tail.tolist(), skip_special_tokens=True)
                out_texts[b0 + i] = text.strip()
            del gen, ids, am
            torch.cuda.empty_cache()
        return out_texts

    # ------------------------------------------------------ patch forward
    @torch.no_grad()
    def patched_logits_at_first_gen(
        self,
        rows: list[dict[str, Any]],
        patch_layer: int,
    ) -> np.ndarray:
        """Row-chunked forward; each row = {'enc': PromptEncoding, 'z': cuda
        tensor (model working dtype)} patches rows[k].enc.decision_pos with
        rows[k].z before patch_layer.  Chunked with OOM-halving because the
        [R, maxlen, V] logits tensor for V=262144 (Gemma-3-4B) is huge.
        Returns first-gen logits [n_rows, V] (float32 cpu)."""
        n = len(rows)
        out_rows: list[np.ndarray] = []
        cur_batch = 12
        b0 = 0
        while b0 < n:
            chunk = rows[b0 : b0 + cur_batch]
            maxlen = max(len(r["enc"].full) for r in chunk)
            ids = torch.full((len(chunk), maxlen), self.pad, dtype=torch.long)
            am = torch.zeros((len(chunk), maxlen), dtype=torch.long)
            for k, r in enumerate(chunk):
                ids[k, : len(r["enc"].full)] = torch.tensor(r["enc"].full, dtype=torch.long)
                am[k, : len(r["enc"].full)] = 1
            ids, am = ids.to(DEVICE), am.to(DEVICE)
            fgp = [r["enc"].first_gen_pos for r in chunk]
            dp = [r["enc"].decision_pos for r in chunk]
            zs = [r["z"] for r in chunk]
            fired = {"n": 0}

            def hook_fn(module, args):
                x = args[0].clone()
                for k in range(x.shape[0]):
                    x[k, dp[k]] = zs[k]
                fired["n"] += 1
                return x

            hook = self._text_layer_module(patch_layer).register_forward_pre_hook(hook_fn)
            try:
                try:
                    out = self.model(input_ids=ids, attention_mask=am, use_cache=False,
                                     output_hidden_states=False)
                    lg = out.logits.float()
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if cur_batch > 1:
                        cur_batch = max(1, cur_batch // 2)
                        logger.warning(
                            f"patched_logits_at_first_gen OOM: halved batch to "
                            f"{cur_batch} for {self.model_id}"
                        )
                        del ids, am
                        continue
                    raise
            finally:
                hook.remove()
            assert fired["n"] >= 1, "patch hook never fired"
            seg = lg[torch.arange(len(chunk)), torch.tensor(fgp, dtype=torch.long)].cpu().numpy()
            out_rows.append(seg)
            del out, lg, ids, am
            b0 += len(chunk)
        return np.concatenate(out_rows, axis=0)

    def free(self) -> None:
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"freed {self.model_id}")


def _cache_has_snapshot(model_id: str) -> bool:
    """True if the HF cache already holds a snapshot of this repo."""
    from pathlib import Path
    import os
    org, _, name = model_id.partition("/")
    hub = Path(os.environ.get("HF_HOME", "hf_cache")) / "hub"
    d = hub / f"models--{org}--{name}" / "snapshots"
    return d.is_dir() and any(p.is_dir() for p in d.iterdir())


# Models that MUST be evaluated in bf16: fp16 overflows in the residual stream
# from about layer 6 onward (verified: google/gemma-3-4b-it with fp16 produces
# NaN hidden states at layers >= 6 and NaN logits; bf16 gives fully finite H,
# d and logits).  Root cause: bf16-trained Gemma3 text model with scaled
# embeddings + linear rope scaling; intermediate magnitudes exceed the fp16
# exponent range.  bf16 numerics are otherwise identical to fp16 for all other
# in-zoo/never-run models (their anchors reproduced iteration-1 exactly).
BF16_REQUIRED = {"google/gemma-3-4b-it", "unsloth/gemma-3-4b-it"}


def load_model(model_id: str, family: str, cls: str, torch_dtype: torch.dtype = torch.float16) -> ModelHandle:
    """Load a model + tokenizer with low CPU memory usage (always anonymous token).

    Iteration-2 addition: gated primaries (meta-llama/..., google/...) whose
    snapshots ARE cached must not hit the hub (token=False -> 401 for gated
    repos), so the load first tries local_files_only=True against the cache and
    only falls back to the hub download path if the cache is incomplete.  The
    numeric definitions are unchanged (fp16, low_cpu_mem_usage, anonymous token).
    """
    if model_id in BF16_REQUIRED:
        # Numerically-necessary dtype override (see module docstring above);
        # logged at INFO so the deviation is never silent.
        logger.info(f"{model_id}: bf16 REQUIRED (fp16 overflows residual stream); overriding dtype")
        torch_dtype = torch.bfloat16
    base_kwargs = dict(trust_remote_code=False, token=False)
    attempts: list[tuple[str, dict]] = [("cache", {"local_files_only": True})] if _cache_has_snapshot(model_id) else []
    attempts.append(("hub", {}))
    last_err: Exception | None = None
    tok = model = None
    for label, kw in attempts:
        try:
            tok = AutoTokenizer.from_pretrained(model_id, **base_kwargs, **kw)
            # Gemma-3-4B (V=262144, 4.3B params fp16 ~= 8.6 GB) does NOT fit by
            # loading to CPU then .to(DEVICE) on a 16 GB GPU (double-buffer OOM,
            # seen as torch.OutOfMemoryError inside .to).  device_map="cuda"
            # streams each shard straight onto the GPU (low_cpu_mem_usage +
            # accelerate), which peaks at ~model_size instead of 2x.
            _lkw = dict(
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                ignore_mismatched_sizes=True,  # e.g. lunahr/gemma-3-1b-it-abliterated has 262145 vs 262144
            )
            if DEVICE.type == "cuda":
                _lkw["device_map"] = "cuda"
            model = AutoModelForCausalLM.from_pretrained(model_id, **_lkw, **base_kwargs, **kw)
            logger.info(f"{model_id}: loaded from {'cache (local_files_only)' if label == 'cache' else 'hub'}")
            break
        except Exception as e:  # noqa: BLE001 - try next source, keep last error
            last_err = e
            logger.warning(f"{model_id}: {label} load failed ({type(e).__name__}: {str(e)[:140]}); trying next source")
            continue
    if tok is None or model is None:
        raise RuntimeError(f"cannot load {model_id} from cache or hub: {last_err}") from last_err
    if DEVICE.type == "cuda":
        if next(model.parameters()).device.type != "cuda":
            model = model.to(DEVICE)
    else:
        model = model.to(DEVICE)
    model.eval()
    # match embedding size to the tokenizer (word-embedding mismatches e.g. +1 token)
    if len(tok) != model.get_input_embeddings().num_embeddings:
        logger.warning(f"{model_id}: resizing embeddings {model.get_input_embeddings().num_embeddings} -> {len(tok)}")
        model.resize_token_embeddings(len(tok))
    return ModelHandle(model, tok, family, model_id, cls)


def load_random_control(family: str = "qwen3", torch_dtype: torch.dtype = torch.float16) -> ModelHandle:
    """Random-weights control: Qwen3-0.6B-Instruct config, default init."""
    from transformers import AutoConfig
    from transformers.models.qwen3 import Qwen3ForCausalLM
    cfg = AutoConfig.from_pretrained("rd211/Qwen3-0.6B-Instruct", token=False)
    model = Qwen3ForCausalLM(cfg)
    model = model.to(dtype=torch_dtype).to(DEVICE).eval()
    tok = AutoTokenizer.from_pretrained("rd211/Qwen3-0.6B-Instruct", token=False)
    return ModelHandle(model, tok, family, "random-weights-control", "random",
                       base_cfg={"config": "rd211/Qwen3-0.6B-Instruct (mirror)"})


def layer_grid(L: int) -> list[int]:
    """Evenly spaced ceil(L/4) patch layers in 1..L-1 including the last."""
    k = max(2, math.ceil(L / 4))
    if k >= L:
        k = min(L - 1, 2)
    pts = np.round(np.linspace(1, L - 1, k)).astype(int).tolist()
    return sorted(set(pts))