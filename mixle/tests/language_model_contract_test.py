"""Token-integrity, artifact, generation, and accounting contracts for LM."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mixle.models.language_model import LM, LMFitReceipt  # noqa: E402


def _small_lm() -> LM:
    torch.manual_seed(0)
    return LM(vocab=11, d_model=4, n_layer=1, n_head=1, block=4)


@pytest.mark.parametrize(
    "tokens",
    [
        [1.9],
        [True],
        [[1]],
        np.asarray([1.0, 2.0]),
        torch.asarray([1.0, 2.0]),
    ],
)
def test_token_validation_rejects_nonintegral_or_nonscalar_ids(tokens: object) -> None:
    with pytest.raises((TypeError, ValueError), match="token id"):
        _small_lm()._check_ids(tokens, "test")


def test_token_validation_consumes_one_shot_iterable_once_into_immutable_int64() -> None:
    result = _small_lm()._check_ids((token for token in [1, 2, 3]), "test")
    np.testing.assert_array_equal(result, np.asarray([1, 2, 3], dtype=np.int64))
    assert result.dtype == np.int64
    assert result.flags.writeable is False


@pytest.mark.parametrize("distributed", [False, True])
def test_streaming_fit_passes_validated_corpus_to_every_backend(
    monkeypatch: pytest.MonkeyPatch,
    distributed: bool,
) -> None:
    lm = _small_lm()
    captured: list[np.ndarray] = []
    source_module = __import__("mixle.data.stream_token_source", fromlist=["stream_token_source"])
    streaming_module = __import__(
        "mixle.models.streaming_transformer_leaf",
        fromlist=["StreamingTransformerLeafEstimator", "stream_fit"],
    )
    sequence_module = __import__("mixle.stats.compute.sequence", fromlist=["seq_estimate"])
    parallel_module = __import__("mixle.utils.parallel.torch_neural", fromlist=["StreamingTokenEncodedData"])

    if distributed:

        class CaptureHandle:
            def __init__(self, token_ids: object, **kwargs: object) -> None:
                captured.append(np.asarray(token_ids).copy())

        monkeypatch.setattr(parallel_module, "StreamingTokenEncodedData", CaptureHandle)
        monkeypatch.setattr(
            sequence_module,
            "seq_estimate",
            lambda handle, estimator, estimate: SimpleNamespace(module=lm.module),
        )
    else:
        monkeypatch.setattr(
            source_module,
            "stream_token_source",
            lambda token_ids, **kwargs: captured.append(np.asarray(token_ids).copy()) or [],
        )
        monkeypatch.setattr(
            streaming_module,
            "stream_fit",
            lambda module, source, **kwargs: (SimpleNamespace(module=module), None),
        )

    lm.fit((token for token in [1, 2, 3, 4, 5]), epochs=1, distributed=distributed)
    assert len(captured) == 1
    np.testing.assert_array_equal(captured[0], np.asarray([1, 2, 3, 4, 5], dtype=np.int64))
    assert captured[0].dtype == np.int64


def test_to_dict_is_detached_cloned_cpu_snapshot() -> None:
    lm = _small_lm()
    snapshot = lm.to_dict()
    name, parameter = next(iter(lm.module.state_dict().items()))
    before = snapshot["state_dict"][name].clone()

    with torch.no_grad():
        parameter.add_(1.0)

    torch.testing.assert_close(snapshot["state_dict"][name], before)
    assert snapshot["state_dict"][name].device.type == "cpu"
    assert snapshot["state_dict"][name].data_ptr() != parameter.data_ptr()


def test_safe_load_fails_closed_when_weights_only_is_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def unsupported(path: str, **kwargs: object) -> object:
        calls.append(kwargs)
        raise TypeError("weights_only unsupported")

    monkeypatch.setattr(torch, "load", unsupported)
    with pytest.raises(RuntimeError, match="safe LM loading"):
        LM.load("artifact.pt")
    assert calls == [{"weights_only": True}]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n": -1}, "n"),
        ({"n": 1.5}, "n"),
        ({"temperature": 0.0}, "temperature"),
        ({"temperature": np.nan}, "temperature"),
        ({"greedy": 1}, "greedy"),
        ({"seed": -1}, "seed"),
    ],
)
def test_generate_rejects_invalid_controls(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _small_lm().generate([1], **kwargs)


def test_generate_rejects_empty_prompt_and_restores_exact_modes_with_int64_input() -> None:
    lm = _small_lm()
    with pytest.raises(ValueError, match="non-empty prompt"):
        lm.generate([], n=1)

    lm.module.train()
    lm.module.blocks[0].mlp.eval()
    modes = {module: module.training for module in lm.module.modules()}
    seen_dtypes: list[torch.dtype] = []
    handle = lm.module.register_forward_pre_hook(lambda module, args: seen_dtypes.append(args[0].dtype))
    try:
        result = lm.generate([1, 2], n=1, greedy=True)
    finally:
        handle.remove()

    assert len(result) == 3
    assert seen_dtypes == [torch.int64]
    assert all(module.training == modes[module] for module in lm.module.modules())


def test_nll_preserves_large_token_ids_as_int64(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[tuple[np.ndarray, np.ndarray]] = []
    streaming_module = __import__(
        "mixle.models.streaming_transformer_leaf",
        fromlist=["StreamingTransformerLeaf"],
    )

    class CaptureLeaf:
        def __init__(self, module: object, device: object) -> None:
            pass

        def seq_log_density(self, encoded: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
            contexts, targets = encoded
            captured.append((contexts.copy(), targets.copy()))
            return np.zeros(len(targets))

    monkeypatch.setattr(streaming_module, "StreamingTransformerLeaf", CaptureLeaf)
    lm = object.__new__(LM)
    lm.vocab = 2**24 + 100
    lm.block = 2
    lm.module = object()
    lm.device = "cpu"
    ids = np.asarray([2**24 + 1, 2**24 + 2, 2**24 + 3, 2**24 + 4], dtype=np.int64)

    assert lm.nll(ids) == 0.0
    assert captured[0][0].dtype == np.int64
    assert captured[0][1].dtype == np.int64
    np.testing.assert_array_equal(captured[0][0][0], ids[:2])


def test_dense_fit_receipt_reports_consumed_supervised_and_dropped_tokens() -> None:
    lm = _small_lm()
    ids = np.arange(12, dtype=np.int64) % lm.vocab
    lm.fit(ids, dense=True, epochs=0, shuffle=False)
    receipt = lm.last_fit_receipt

    assert isinstance(receipt, LMFitReceipt)
    assert receipt.input_tokens == 12
    assert receipt.consumed_tokens_per_epoch == 10
    assert receipt.supervised_tokens_per_epoch == 8
    assert receipt.discarded_cross_row_transitions_per_epoch == 1
    assert receipt.dropped_tail_tokens == 2
    assert receipt.epochs == 0
