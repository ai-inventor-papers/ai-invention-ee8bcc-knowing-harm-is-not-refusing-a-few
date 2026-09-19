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
# continuation scoring materializes [R, len, V] logits at the LM head (V ~ 151k):
# keep the per-forward row count small enough for ~20GB VRAM.
CONT_SCORE_CHUNK = 48


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
        self.L: int = int(getattr(cfg, "num_hidden_layers", 0))
        self.D: int = int(getattr(cfg, "hidden_size", 0))
        self.eos = tok.eos_token_id
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id
        self.pad = tok.pad_token_id
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

    # ----------------------------------------------------------- forwards
    @torch.no_grad()
    def forward_states(self, encs: list[PromptEncoding], batch_size: int = 12) -> dict[str, Any]:
        """One forward per batch; returns post-layer hidden states at decision
        positions, first-gen logits, and the ids tensors."""
        n = len(encs)
        L, D = self.L, self.D
        H = np.zeros((L, n, D), dtype=np.float32)      # H[l] = after layer l (0-based l)
        first_logits: list[np.ndarray] = []
        for b0 in range(0, n, batch_size):
            batch = encs[b0 : b0 + batch_size]
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
        for b0 in range(0, len(rows), CONT_SCORE_CHUNK):
            chunk = rows[b0 : b0 + CONT_SCORE_CHUNK]
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
                hook = self.model.model.layers[patch_layer].register_forward_pre_hook(
                    _make_hook(zs, dp, fired)
                )
            try:
                out = self.model(input_ids=ids, attention_mask=am, use_cache=False, output_hidden_states=False)
                lg = out.logits.float()  # [R, maxlen, V]
            finally:
                if hook is not None:
                    hook.remove()
            if zmap is not None:
                # the pre-forward hook fires once per forward pass (one batch here)
                assert fired["n"] >= 1, f"patch hook never fired"

            # score tokens of c[1:] at appended positions
            seql = torch.logsumexp(lg, dim=-1)  # [R, maxlen]
            for r, (i, j) in enumerate(chunk):
                ids_c = c0_ids[j]
                base = len(encs[i].full)
                acc = 0.0
                for k, tok_id in enumerate(ids_c[1:]):
                    pos = base + k
                    acc += float((lg[r, pos, tok_id] - seql[r, pos]).item())
                logp[i, j] += acc
            del out, lg, ids, am

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
        """One forward; each row = {'enc': PromptEncoding, 'z': cuda fp16 vector}
        patches rows[k].enc.decision_pos with rows[k].z before patch_layer.
        Returns first-gen logits [n_rows, V] (float32 cpu)."""
        n = len(rows)
        maxlen = max(len(r["enc"].full) for r in rows)
        ids = torch.full((n, maxlen), self.pad, dtype=torch.long)
        am = torch.zeros((n, maxlen), dtype=torch.long)
        for k, r in enumerate(rows):
            ids[k, : len(r["enc"].full)] = torch.tensor(r["enc"].full, dtype=torch.long)
            am[k, : len(r["enc"].full)] = 1
        ids, am = ids.to(DEVICE), am.to(DEVICE)
        fgp = [r["enc"].first_gen_pos for r in rows]
        dp = [r["enc"].decision_pos for r in rows]
        zs = [r["z"] for r in rows]
        fired = {"n": 0}

        def hook_fn(module, args):
            x = args[0].clone()
            for k in range(x.shape[0]):
                x[k, dp[k]] = zs[k]
            fired["n"] += 1
            return x

        hook = self.model.model.layers[patch_layer].register_forward_pre_hook(hook_fn)
        try:
            out = self.model(input_ids=ids, attention_mask=am, use_cache=False, output_hidden_states=False)
            lg = out.logits.float()
        finally:
            hook.remove()
        # pre-forward hook fires once per forward pass (one batch here)
        assert fired["n"] >= 1, "patch hook never fired"
        res = lg[torch.arange(n), torch.tensor(fgp, dtype=torch.long)].cpu().numpy()
        del out, lg, ids, am
        return res

    def free(self) -> None:
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"freed {self.model_id}")


def load_model(model_id: str, family: str, cls: str, torch_dtype: torch.dtype = torch.float16) -> ModelHandle:
    """Load a model + tokenizer with low CPU memory usage (always anonymous token)."""
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False, token=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
        token=False,
        ignore_mismatched_sizes=True,  # e.g. lunahr/gemma-3-1b-it-abliterated has 262145 vs 262144
    )
    model = model.to(DEVICE).eval()
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