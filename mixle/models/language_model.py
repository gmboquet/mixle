"""``LM`` -- a declarative autoregressive language model with fit, generation, and scoring helpers.

A causal Transformer trained on a token stream::

    lm = LM(vocab=V, d_model=256, n_layer=6, n_head=8, block=128)
    lm.fit(token_ids, epochs=3, batch_size=64, device="mps")          # pretrain (single process)
    lm.fit(token_ids, dense=True, distributed=True, precision="bf16") # packed DDP/FSDP2 training
    text = lm.generate(prompt_ids, n=200, temperature=0.8)            # autoregressive sampling
    nll  = lm.nll(held_out_ids)                                       # bits/token on held-out data

``fit`` retains the non-buffering streaming estimator for compatibility. Packed
``dense=True`` training uses the distributed-gradient backend when requested,
with deterministic data sharding, microbatches, accumulation, and complete DCP
training state. ``fit_pairs`` is the SFT stage: dense all-position teacher
forcing on ``(prompt, completion)`` pairs with the loss masked to completions.
The rest of the multi-stage pipeline (CPT-with-EWC, DPO) is ``mixle.models.continual`` / ``mixle.models.dpo_leaf``.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _torch() -> Any:
    import torch

    return torch


def _forward_all_positions(module: Any, x: Any, *, position_ids: Any = None) -> Any:
    """Next-token logits at EVERY position, ``(batch, block, vocab)``.

    ``build_causal_lm``'s forward returns only the last position (the shape the leaf estimators score);
    dense teacher forcing needs them all, so this re-runs the same layers off the module's own parts.
    """
    try:
        return module(x, position_ids=position_ids, return_all_logits=True)
    except TypeError as error:
        if "return_all_logits" not in str(error) and "unexpected keyword" not in str(error):
            raise
        # Keep externally supplied CausalLM-compatible modules working.
        torch = _torch()
        t = x.shape[1]
        pos = torch.arange(t, device=x.device) if position_ids is None else position_ids
        position_embeddings = module.pos(pos)
        if position_embeddings.ndim == 2:
            position_embeddings = position_embeddings[None, :, :]
        h = module.tok(x.long()) + position_embeddings
        for blk in module.blocks:
            h = blk(h)
        return module.head(module.ln(h))


class LM:
    """A causal-Transformer language model with a small declarative surface: ``fit`` / ``generate`` / ``nll``."""

    def __init__(
        self,
        vocab: int,
        *,
        d_model: int = 256,
        n_layer: int = 6,
        n_head: int = 8,
        block: int = 128,
        device: str = "cpu",
        embedding: Any = None,
    ) -> None:
        from mixle.models.transformer import build_causal_lm

        self.vocab = int(vocab)
        self.d_model = int(d_model)
        self.n_layer = int(n_layer)
        self.n_head = int(n_head)
        self.block = int(block)
        self.device = device
        # embedding=CategoricalEmbedding ties one word embedding across LMs (e.g. a mixture's per-cluster experts)
        self.module = build_causal_lm(self.vocab, d_model, n_layer, n_head, self.block, embedding=embedding)

    def _check_ids(self, ids: Any, where: str) -> np.ndarray:
        """Validate that every token id is a nonnegative int below ``vocab``; raise naming the offending id."""
        arr = np.asarray([int(t) for t in ids], dtype=np.int64)
        if arr.size:
            bad = arr[(arr < 0) | (arr >= self.vocab)]
            if bad.size:
                raise ValueError("%s: token id %d is outside the vocabulary [0, %d)" % (where, int(bad[0]), self.vocab))
        return arr

    def to_dict(self) -> dict:
        """Serialize the hyperparameters + trained weights so the LM survives a process boundary.

        The token embedding may be tied across LMs (``embedding=``); ``from_dict`` rebuilds an untied module and
        loads the saved ``state_dict`` into it, so a round-tripped LM is standalone (any external tie is dropped).
        """
        torch = _torch()
        state = {k: v.cpu() for k, v in self.module.state_dict().items()}
        return {
            "vocab": self.vocab,
            "d_model": self.d_model,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "block": self.block,
            "device": self.device,
            "state_dict": state,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> LM:
        """Rebuild an LM from :meth:`to_dict` output (fresh module, saved weights loaded in)."""
        lm = cls(
            vocab=payload["vocab"],
            d_model=payload["d_model"],
            n_layer=payload["n_layer"],
            n_head=payload["n_head"],
            block=payload["block"],
            device=payload.get("device", "cpu"),
        )
        lm.module.load_state_dict(payload["state_dict"])
        return lm

    def save(self, path: str) -> None:
        """Persist the LM's **inference artifact** to ``path`` via ``torch.save``: architecture config
        (vocab/d_model/n_layer/n_head/block) + learned weights, nothing else.

        This is **not** a training checkpoint. It deliberately does not store the optimizer, LR scheduler,
        step count, RNG state, data-loader position, or gradient-scaler state, so :meth:`load` returns a
        model ready for scoring and generation, not one you can resume training from without loss of state.
        For resumable training use the checkpoint path in :mod:`mixle.utils.parallel.fault_tolerant_training`
        / :mod:`mixle.utils.parallel.dcp_checkpoint`, which stores model + optimizer + data-loader position.
        """
        _torch().save(self.to_dict(), path)

    @classmethod
    def load(cls, path: str) -> LM:
        """Load an LM **inference artifact** written by :meth:`save` (config + weights).

        Returns a model ready for scoring/generation; this is not a training-resume checkpoint (see
        :meth:`save`), so optimizer/step/RNG/data-loader state is not restored.

        Loaded with ``weights_only=True``: :meth:`to_dict`'s payload is plain config scalars plus a
        tensor ``state_dict``, never an arbitrary object graph, so this does not execute code from the
        file (unlike a full-module pickle -- see ``mixle.models._neural_serial`` for that case).
        """
        torch = _torch()
        try:
            payload = torch.load(path, weights_only=True)
        except TypeError:  # torch < 1.13 has no weights_only kwarg
            payload = torch.load(path)
        return cls.from_dict(payload)

    def fit(
        self,
        token_ids: Any,
        *,
        epochs: int = 1,
        batch_size: int = 64,
        lr: float = 3e-3,
        distributed: bool = False,
        precision: str = "fp32",
        shuffle: bool = True,
        dense: bool = False,
        seed: int = 0,
        log: Any = None,
        tp_size: int = 1,
        pp_size: int = 1,
        cp_size: int = 1,
        dp_replicate: int = 1,
        dp_shard: int = 1,
        ep_size: int = 1,
        etp_size: int = 1,
        microbatches: int = 1,
        gradient_accumulation_steps: int = 1,
        distributed_backend: str = "torch_native",
        optimizer: Any = None,
        max_grad_norm: float | None = None,
        compile: bool = False,
        checkpoint_path: str | None = None,
        resume: bool = False,
    ) -> LM:
        """Small-scale reference pretraining (or continuation) on a token-id array.

        Two paths, chosen by ``dense``:

        - ``dense=False`` (default, unchanged): the streaming estimator; the corpus is never
          buffered. It scores one next-token target per ``block``-length window -- the right shape
          for an unbounded stream, but only a fraction (~``1/block``) of the token supervision that
          packed dense teacher forcing extracts from the same tokens. A reference / continuation
          path for small-scale runs, not a frontier pretraining engine.
        - ``dense=True``: packed dense teacher forcing on a BOUNDED corpus. The token stream is cut
          into non-overlapping rows of ``block + 1`` tokens (a partial tail row is dropped; there is
          no padding, so no attention-mask or EOS special-casing is needed -- document boundaries are
          whatever the ids encode); every row contributes ``block`` shifted next-token targets in a
          single forward, recovering the ~``block``x supervision the streaming path leaves unused.
          Token ids stay integer end to end (no float conversion). ``seed`` drives row shuffling and
          ``log(epoch, mean_loss)`` reports per-epoch training loss, both as in :meth:`fit_pairs`.
          With ``distributed=True`` this objective is executed by the selected
          distributed-gradient backend.

        Parallel sizes are executable contracts. ``torch_native`` supports
        DDP/HSDP, FSDP2, MLP tensor parallelism, and CUDA context parallelism;
        unsupported combinations fail capability validation. Full TP/PP/CP/EP
        transformer and MoE training is exposed by the ``megatron`` backend.
        """
        ids = self._check_ids(token_ids, "fit")
        if dense:
            if distributed:
                return self._fit_dense_distributed(
                    ids,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    precision=precision,
                    shuffle=shuffle,
                    seed=seed,
                    log=log,
                    tp_size=tp_size,
                    pp_size=pp_size,
                    cp_size=cp_size,
                    dp_replicate=dp_replicate,
                    dp_shard=dp_shard,
                    ep_size=ep_size,
                    etp_size=etp_size,
                    microbatches=microbatches,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    distributed_backend=distributed_backend,
                    optimizer=optimizer,
                    max_grad_norm=max_grad_norm,
                    compile=compile,
                    checkpoint_path=checkpoint_path,
                    resume=resume,
                )
            return self._fit_dense(
                ids, epochs=epochs, batch_size=batch_size, lr=lr, shuffle=shuffle, seed=seed, log=log
            )
        if distributed:
            requested_model_axes = {
                "dp_replicate": dp_replicate,
                "dp_shard": dp_shard,
                "tp_size": tp_size,
                "pp_size": pp_size,
                "cp_size": cp_size,
                "ep_size": ep_size,
                "etp_size": etp_size,
            }
            active = [name for name, size in requested_model_axes.items() if int(size) > 1]
            if active:
                raise ValueError(
                    "the streaming one-target objective does not execute an explicit parallel plan (%s); "
                    "use dense=True with torch_native where supported, or a Megatron provider." % ", ".join(active)
                )
            from mixle.models.streaming_transformer_leaf import StreamingTransformerLeafEstimator
            from mixle.stats.compute.sequence import seq_estimate
            from mixle.utils.parallel.torch_neural import StreamingTokenEncodedData

            handle = StreamingTokenEncodedData(
                token_ids,
                block=self.block,
                batch_size=batch_size,
                epochs=epochs,
                shuffle=shuffle,
                precision=precision,
                tp_size=tp_size,
                pp_size=pp_size,
                cp_size=cp_size,
            )
            est = StreamingTransformerLeafEstimator(self.module, lr=lr, device=self.device)
            self.module = seq_estimate(handle, est, None).module
        else:
            from mixle.data.stream_token_source import stream_token_source
            from mixle.models.streaming_transformer_leaf import stream_fit

            src = stream_token_source(
                token_ids, block=self.block, batch_size=batch_size, epochs=epochs, shuffle=shuffle
            )
            self.module = stream_fit(self.module, src, lr=lr, device=self.device)[0].module
        return self

    def _fit_dense_distributed(
        self,
        ids: np.ndarray,
        *,
        epochs: int,
        batch_size: int,
        lr: float,
        precision: str,
        shuffle: bool,
        seed: int,
        log: Any,
        tp_size: int,
        pp_size: int,
        cp_size: int,
        dp_replicate: int,
        dp_shard: int,
        ep_size: int,
        etp_size: int,
        microbatches: int,
        gradient_accumulation_steps: int,
        distributed_backend: str,
        optimizer: Any,
        max_grad_norm: float | None,
        compile: bool,
        checkpoint_path: str | None,
        resume: bool,
    ) -> LM:
        """Packed all-position training over an explicit distributed mesh."""

        torch = _torch()
        from mixle.utils.parallel import ParallelPlan, get_training_backend

        stride = self.block + 1
        n_rows = len(ids) // stride
        if n_rows < 1:
            raise ValueError(
                "fit(dense=True) needs at least block+1=%d tokens to form one packed row; got %d" % (stride, len(ids))
            )
        plan = ParallelPlan(
            dp_replicate=dp_replicate,
            dp_shard=dp_shard,
            tp=tp_size,
            pp=pp_size,
            cp=cp_size,
            ep=ep_size,
            etp=etp_size,
            microbatches=microbatches,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        backend = get_training_backend(distributed_backend)
        session = backend.prepare(
            self.module,
            plan=plan,
            device=self.device,
            precision=precision,
            optimizer=optimizer,
            lr=lr,
            max_grad_norm=max_grad_norm,
            compile=compile,
        )
        flat = torch.as_tensor(ids[: n_rows * stride], dtype=torch.int64).view(n_rows, stride)
        try:
            start_epoch = 0
            if resume:
                if checkpoint_path is None:
                    raise ValueError("resume=True requires checkpoint_path.")
                payload = session.load_checkpoint(checkpoint_path)
                loader_state = payload.get("loader_state") or {}
                start_epoch = int(loader_state.get("epoch", 0))
            for epoch in range(start_epoch, int(epochs)):
                epoch_seed = int(np.random.SeedSequence([seed, epoch]).generate_state(1)[0])
                rng = np.random.RandomState(epoch_seed)
                global_order = rng.permutation(n_rows) if shuffle else np.arange(n_rows)
                local_order = global_order[session.data_parallel_rank :: session.data_parallel_size]
                total_loss = 0.0
                total_tokens = 0.0
                for start in range(0, len(local_order), int(batch_size)):
                    index = torch.as_tensor(local_order[start : start + int(batch_size)], dtype=torch.int64)
                    rows = flat[index]
                    receipt = session.train_batch(rows[:, :-1], rows[:, 1:])
                    if not receipt.skipped:
                        total_loss += receipt.loss * receipt.local_tokens
                        total_tokens += receipt.local_tokens
                receipt = session.finish_accumulation()
                if receipt is not None:
                    total_loss += receipt.loss * receipt.local_tokens
                    total_tokens += receipt.local_tokens
                if hasattr(session, "reduce_sums"):
                    total_loss, total_tokens = session.reduce_sums(total_loss, total_tokens)
                if log is not None and session.is_logging_rank:
                    log(epoch, total_loss / max(total_tokens, 1.0))
                if checkpoint_path is not None:
                    session.save_checkpoint(
                        checkpoint_path,
                        loader_state={"seed": seed, "epoch": epoch + 1, "batch": 0},
                        extra={"objective": "packed_causal_lm"},
                    )
        finally:
            session.close()
        wrapped = session.module
        while hasattr(wrapped, "module"):
            wrapped = wrapped.module
        self.module = getattr(wrapped, "_orig_mod", wrapped)
        return self

    def _fit_dense(
        self,
        ids: np.ndarray,
        *,
        epochs: int,
        batch_size: int,
        lr: float,
        shuffle: bool,
        seed: int,
        log: Any,
    ) -> LM:
        """Packed dense teacher forcing: non-overlapping ``block + 1``-token rows, all-position loss.

        Row ``r`` covers ``ids[r*(block+1) : (r+1)*(block+1)]``; inputs are its first ``block`` tokens
        and targets its last ``block`` (shift by one), so every consumed token is supervised exactly
        once and rows carry no padding. A partial tail row is dropped. Rows are gathered from the flat
        id tensor per batch, so nothing beyond one batch is materialized on the device.
        """
        torch = _torch()
        stride = self.block + 1
        n_rows = len(ids) // stride
        if n_rows < 1:
            raise ValueError(
                "fit(dense=True) needs at least block+1=%d tokens to form one packed row; got %d" % (stride, len(ids))
            )
        rng = np.random.RandomState(seed)
        module = self.module.to(self.device)
        module.train()
        opt = torch.optim.Adam(module.parameters(), lr=lr)
        flat = torch.as_tensor(ids[: n_rows * stride], dtype=torch.int64).view(n_rows, stride)
        for epoch in range(int(epochs)):
            order = rng.permutation(n_rows) if shuffle else np.arange(n_rows)
            total = count = 0.0
            for s in range(0, n_rows, int(batch_size)):
                rows = flat[torch.as_tensor(order[s : s + int(batch_size)], dtype=torch.int64)].to(self.device)
                logits = _forward_all_positions(module, rows[:, :-1])
                target = rows[:, 1:]
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, self.vocab), target.reshape(-1))
                opt.zero_grad()
                loss.backward()
                opt.step()
                n_targets = float(target.numel())
                total += float(loss.detach()) * n_targets
                count += n_targets
            if log is not None:
                log(epoch, total / max(count, 1.0))
        return self

    def fit_pairs(
        self,
        pairs: Any,
        *,
        epochs: int = 1,
        batch_size: int = 32,
        lr: float = 3e-3,
        mask_prompt: bool = True,
        pad_id: int = 0,
        seed: int = 0,
        log: Any = None,
    ) -> LM:
        """Supervised fine-tuning on ``(prompt_ids, completion_ids)`` pairs with a dense per-position loss.

        The streaming ``fit`` path scores ONE next-token target per window (the right shape for an unbounded
        pretraining stream); for a pair corpus that wastes a factor of ``block`` in compute. Here every position
        of every pair contributes cross-entropy in a single forward, and ``mask_prompt`` restricts the loss to
        completion positions -- the standard SFT objective. Sequences longer than ``block`` keep the completion
        and drop the oldest prompt tokens; shorter ones are left-padded with ``pad_id`` (excluded from the loss).
        Include your end-of-sequence token in each completion so ``generate(stop_id=...)`` knows where to stop.
        """
        torch = _torch()
        pairs = list(pairs)
        if not pairs:
            return self  # nothing to fine-tune on
        rng = np.random.RandomState(seed)
        rows, tmask = [], []
        for prompt, completion in pairs:
            self._check_ids(prompt, "fit_pairs (prompt)")
            self._check_ids(completion, "fit_pairs (completion)")
            seq = [int(t) for t in prompt] + [int(t) for t in completion]
            keep = [False] * (len(seq) if mask_prompt else 0)
            if mask_prompt:
                for i in range(len(prompt), len(seq)):
                    keep[i] = True
            else:
                keep = [True] * len(seq)
            if len(seq) > self.block:
                seq, keep = seq[-self.block :], keep[-self.block :]
            pad = self.block - len(seq)
            # right-align the pad so tokens sit at positions 0..len-1 -- the same absolute positions
            # generate() feeds them at (it runs the unpadded window), keeping train/decode consistent
            rows.append(seq + [pad_id] * pad)
            tmask.append(keep + [False] * pad)
        x = torch.as_tensor(np.asarray(rows, dtype=np.int64))
        # position t's logits predict token t+1: shift the target/mask left by one
        target = x[:, 1:].to(self.device)
        m = torch.as_tensor(np.asarray(tmask, dtype=bool))[:, 1:].to(self.device)
        x = x.to(self.device)
        module = self.module.to(self.device)
        module.train()
        opt = torch.optim.Adam(module.parameters(), lr=lr)
        n = x.shape[0]
        for epoch in range(int(epochs)):
            order = rng.permutation(n)
            total = count = 0.0
            for s in range(0, n, int(batch_size)):
                idx = torch.as_tensor(order[s : s + int(batch_size)], dtype=torch.int64, device=self.device)
                logits = _forward_all_positions(module, x[idx])[:, :-1]
                mask = m[idx]
                if not bool(mask.any()):
                    continue
                loss = torch.nn.functional.cross_entropy(logits[mask], target[idx][mask])
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss.detach()) * float(mask.sum())
                count += float(mask.sum())
            if log is not None:
                log(epoch, total / max(count, 1.0))
        return self

    def generate(
        self,
        prompt_ids: Any,
        n: int = 200,
        *,
        temperature: float = 1.0,
        greedy: bool = False,
        seed: int = 0,
        stop_id: int | None = None,
    ) -> list:
        """Autoregressively extend ``prompt_ids`` by ``n`` tokens (greedy, or temperature-sampled).

        ``stop_id`` ends generation early when that token is produced (it is included in the return value,
        so callers can strip it -- and its presence distinguishes 'finished' from 'ran out of budget').
        """
        torch = _torch()
        self._check_ids(prompt_ids, "generate")
        if stop_id is not None:
            self._check_ids([stop_id], "generate (stop_id)")
        rng = np.random.RandomState(seed)
        self.module.to(self.device).eval()
        w = [int(t) for t in prompt_ids]
        out = list(w)
        for _ in range(int(n)):
            # feed the window unpadded: positions run 0..len-1 exactly as in training (fit / fit_pairs),
            # and the attention cost tracks the true length instead of always paying the full block
            win = w[-self.block :]
            with torch.no_grad():
                logits = self.module(torch.as_tensor([win], dtype=torch.float32).to(self.device))[0].cpu().numpy()
            if greedy:
                nxt = int(logits.argmax())
            else:
                p = np.exp((logits - logits.max()) / max(temperature, 1e-6))
                p /= p.sum()
                nxt = int(rng.choice(len(p), p=p))
            w.append(nxt)
            out.append(nxt)
            if stop_id is not None and nxt == int(stop_id):
                break
        self.module.train()
        return out

    def nll(self, token_ids: Any) -> float:
        """Mean next-token negative log-likelihood (nats/token) on a token-id array."""
        from mixle.models.streaming_transformer_leaf import StreamingTransformerLeaf

        ids = self._check_ids(token_ids, "nll")
        if len(ids) <= self.block:
            raise ValueError(
                "nll needs more than block=%d tokens to score at least one next-token target; got %d"
                % (self.block, len(ids))
            )
        leaf = StreamingTransformerLeaf(self.module, self.device)
        n = len(ids) - self.block
        # Stream the (n, block) context windows in chunks rather than materializing the whole matrix (which was
        # ~2GB at 1M tokens, block=512); accumulate the summed log-density and divide by the token count.
        total = 0.0
        chunk = 4096
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            ctx = np.stack([ids[i : i + self.block] for i in range(start, stop)]).astype("float32")
            total += float(np.sum(leaf.seq_log_density((ctx, ids[self.block + start : self.block + stop]))))
        return -total / n
