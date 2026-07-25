"""``LM`` -- a declarative autoregressive language model with fit, generation, and scoring helpers.

A causal Transformer trained on a token stream::

    lm = LM(vocab=V, d_model=256, n_layer=6, n_head=8, block=128)
    lm.fit(token_ids, epochs=3, batch_size=64, device="mps")          # pretrain (single process)
    lm.fit(token_ids, dense=True, distributed=True, precision="bf16") # packed DDP/FSDP2 training
    text = lm.generate(prompt_ids, n=200, temperature=0.8)            # autoregressive sampling
    nll  = lm.nll(held_out_ids)                                       # nats/token on held-out data

``fit`` retains the non-buffering streaming estimator for compatibility. Packed
``dense=True`` training uses the distributed-gradient backend when requested,
with deterministic data sharding, microbatches, accumulation, and complete DCP
training state. ``fit_pairs`` is the SFT stage: dense all-position teacher
forcing on ``(prompt, completion)`` pairs with the loss masked to completions.
The rest of the multi-stage pipeline (CPT-with-EWC, DPO) is ``mixle.models.continual`` / ``mixle.models.dpo_leaf``.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np

from mixle.models.grad_leaf import _module_mode


@dataclass(frozen=True)
class LMFitReceipt:
    """Token-accounting receipt for one packed dense corpus layout."""

    objective: str
    input_tokens: int
    consumed_tokens_per_epoch: int
    supervised_tokens_per_epoch: int
    discarded_cross_row_transitions_per_epoch: int
    dropped_tail_tokens: int
    epochs: int


def _torch() -> Any:
    import torch

    return torch


def _exact_int(value: Any, name: str, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _positive_finite(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a positive finite real number")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite real number")
    return result


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
        self.last_fit_receipt: LMFitReceipt | None = None
        # embedding=CategoricalEmbedding ties one word embedding across LMs (e.g. a mixture's per-cluster experts)
        self.module = build_causal_lm(self.vocab, d_model, n_layer, n_head, self.block, embedding=embedding)

    def _check_ids(self, ids: Any, where: str) -> np.ndarray:
        """Validate that every token id is a nonnegative int below ``vocab``; raise naming the offending id."""
        torch = _torch()
        if torch.is_tensor(ids):
            if ids.ndim != 1 or ids.dtype not in (
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            ):
                raise TypeError(f"{where}: token ids must be a one-dimensional sequence of exact integers")
            arr = ids.detach().to(device="cpu", dtype=torch.int64).numpy().copy()
        else:
            try:
                values = list(ids)
            except TypeError as exc:
                raise TypeError(f"{where}: token ids must be an iterable of exact integers") from exc
            for index, token in enumerate(values):
                if isinstance(token, (bool, np.bool_)) or not isinstance(token, Integral):
                    raise TypeError(f"{where}: token id at index {index} must be an exact integer; got {token!r}")
            arr = np.asarray(values, dtype=np.int64)
        if arr.ndim != 1:
            raise ValueError(f"{where}: token ids must be one-dimensional")
        if arr.size:
            bad = arr[(arr < 0) | (arr >= self.vocab)]
            if bad.size:
                raise ValueError("%s: token id %d is outside the vocabulary [0, %d)" % (where, int(bad[0]), self.vocab))
        arr.setflags(write=False)
        return arr

    def to_dict(self) -> dict:
        """Serialize the hyperparameters + trained weights so the LM survives a process boundary.

        The token embedding may be tied across LMs (``embedding=``); ``from_dict`` rebuilds an untied module and
        loads the saved ``state_dict`` into it, so a round-tripped LM is standalone (any external tie is dropped).
        """
        torch = _torch()
        state = {k: v.detach().to(device="cpu").clone() for k, v in self.module.state_dict().items()}
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
        except TypeError as exc:  # pragma: no cover - supported Torch versions provide safe loading
            raise RuntimeError(
                "safe LM loading requires torch.load(..., weights_only=True); "
                "upgrade Torch rather than falling back to executable pickle loading"
            ) from exc
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
          whatever the ids encode). Every row consumes ``block+1`` tokens but contributes ``block``
          shifted targets: the corpus's first token and each cross-row transition are not supervised,
          and the partial tail is not consumed. :attr:`last_fit_receipt` reports those counts
          explicitly. Token ids stay integer end to end (no float conversion). ``seed`` drives row
          shuffling and ``log(epoch, mean_loss)`` reports per-epoch training loss, both as in
          :meth:`fit_pairs`. With ``distributed=True`` this objective is executed by the selected
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
                ids,
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

            src = stream_token_source(ids, block=self.block, batch_size=batch_size, epochs=epochs, shuffle=shuffle)
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
        flat = torch.tensor(ids[: n_rows * stride], dtype=torch.int64).view(n_rows, stride)
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
        consumed = n_rows * stride
        self.last_fit_receipt = LMFitReceipt(
            objective="packed_causal_lm",
            input_tokens=len(ids),
            consumed_tokens_per_epoch=consumed,
            supervised_tokens_per_epoch=n_rows * self.block,
            discarded_cross_row_transitions_per_epoch=max(0, n_rows - 1),
            dropped_tail_tokens=len(ids) - consumed,
            epochs=int(epochs),
        )
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
        and targets its last ``block`` (shift by one). Thus each row consumes ``block+1`` tokens and
        supervises ``block`` targets. The first corpus token is context-only, transitions between rows
        are discarded, and a partial tail row is dropped. :attr:`last_fit_receipt` records each count.
        Rows are gathered from the flat id tensor per batch, so nothing beyond one batch is materialized
        on the device.
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
        flat = torch.tensor(ids[: n_rows * stride], dtype=torch.int64).view(n_rows, stride)
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
        consumed = n_rows * stride
        self.last_fit_receipt = LMFitReceipt(
            objective="packed_causal_lm",
            input_tokens=len(ids),
            consumed_tokens_per_epoch=consumed,
            supervised_tokens_per_epoch=n_rows * self.block,
            discarded_cross_row_transitions_per_epoch=max(0, n_rows - 1),
            dropped_tail_tokens=len(ids) - consumed,
            epochs=int(epochs),
        )
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
        pad_id = int(self._check_ids([pad_id], "fit_pairs (pad_id)")[0])
        rng = np.random.RandomState(seed)
        rows, tmask = [], []
        for prompt, completion in pairs:
            prompt_array = self._check_ids(prompt, "fit_pairs (prompt)")
            completion_array = self._check_ids(completion, "fit_pairs (completion)")
            seq = prompt_array.tolist() + completion_array.tolist()
            keep = [False] * (len(seq) if mask_prompt else 0)
            if mask_prompt:
                for i in range(len(prompt_array), len(seq)):
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
        prompt = self._check_ids(prompt_ids, "generate")
        if not len(prompt):
            raise ValueError("generate requires a non-empty prompt")
        n = _exact_int(n, "n", minimum=0)
        temperature = _positive_finite(temperature, "temperature")
        if not isinstance(greedy, (bool, np.bool_)):
            raise TypeError("greedy must be a boolean")
        seed = _exact_int(seed, "seed", minimum=0)
        stop_token: int | None = None
        if stop_id is not None:
            stop_token = int(self._check_ids([stop_id], "generate (stop_id)")[0])
        rng = np.random.RandomState(seed)
        self.module.to(self.device)
        w = prompt.tolist()
        out = list(w)
        with _module_mode(self.module, train=False), torch.no_grad():
            for _ in range(n):
                # Feed the window unpadded: positions run 0..len-1 exactly as in training, and token IDs
                # remain int64 so values above 2**24 are never rounded through float32.
                win = w[-self.block :]
                x = torch.as_tensor([win], dtype=torch.int64, device=self.device)
                raw_logits = self.module(x)
                if (
                    not isinstance(raw_logits, torch.Tensor)
                    or tuple(raw_logits.shape) != (1, self.vocab)
                    or not bool(torch.isfinite(raw_logits).all())
                ):
                    raise ValueError(
                        f"generate module logits must be a finite (1, {self.vocab}) matrix; "
                        f"got {getattr(raw_logits, 'shape', None)}"
                    )
                logits = raw_logits[0].detach().to(device="cpu", dtype=torch.float64).numpy()
                if greedy:
                    nxt = int(logits.argmax())
                else:
                    probabilities = np.exp((logits - logits.max()) / temperature)
                    normalizer = float(probabilities.sum())
                    if not np.all(np.isfinite(probabilities)) or not np.isfinite(normalizer) or normalizer <= 0.0:
                        raise ValueError("generate produced invalid sampling probabilities")
                    probabilities /= normalizer
                    nxt = int(rng.choice(len(probabilities), p=probabilities))
                w.append(nxt)
                out.append(nxt)
                if stop_token is not None and nxt == stop_token:
                    break
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
            ctx = np.stack([ids[i : i + self.block] for i in range(start, stop)]).astype(np.int64, copy=False)
            total += float(np.sum(leaf.seq_log_density((ctx, ids[self.block + start : self.block + stop]))))
        return -total / n
