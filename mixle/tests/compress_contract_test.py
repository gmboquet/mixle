"""Adverse lifecycle, selection-data, budget, and receipt contracts for compression."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import mixle.models.compress as compression  # noqa: E402
from mixle.models.compress import MethodCandidate, _quality, _trainable_blocks, compress  # noqa: E402
from mixle.models.moment_propagation import GaussianLaw  # noqa: E402
from mixle.models.transformer import build_causal_lm  # noqa: E402


def _small_model() -> object:
    torch.manual_seed(0)
    return build_causal_lm(vocab=9, d_model=4, n_layer=2, n_head=1, block=4)


def _input_law() -> GaussianLaw:
    return GaussianLaw(mu=np.zeros(4), covar=np.eye(4))


def test_trainable_block_context_restores_every_original_flag_after_exception() -> None:
    model = _small_model()
    selected_parameters = list(model.blocks[0].parameters())
    selected_parameters[0].requires_grad_(False)
    original = {id(parameter): parameter.requires_grad for parameter in model.parameters()}

    with pytest.raises(RuntimeError, match="forced"):
        with _trainable_blocks(model, [0]):
            assert selected_parameters[0].requires_grad is False
            assert any(parameter.requires_grad for parameter in selected_parameters[1:])
            assert not any(parameter.requires_grad for parameter in model.blocks[1].parameters())
            raise RuntimeError("forced")

    assert all(parameter.requires_grad == original[id(parameter)] for parameter in model.parameters())


def test_quality_preserves_modes_and_rejects_empty_or_nonfinite_results() -> None:
    teacher = _small_model()
    student = copy.deepcopy(teacher)
    teacher.train()
    teacher.blocks[0].mlp.eval()
    student.eval()
    student.blocks[1].train()
    teacher_modes = {module: module.training for module in teacher.modules()}
    student_modes = {module: module.training for module in student.modules()}
    evaluation = torch.randint(0, teacher.vocab, (3, teacher.block))

    assert _quality(student, teacher, evaluation) == pytest.approx(1.0)
    assert all(module.training == teacher_modes[module] for module in teacher.modules())
    assert all(module.training == student_modes[module] for module in student.modules())

    with pytest.raises(ValueError, match="non-empty"):
        _quality(student, teacher, torch.empty((0, teacher.block), dtype=torch.long))

    class NonFiniteModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.vocab = teacher.vocab
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.full(
                (x.shape[0], self.vocab),
                np.nan,
                device=x.device,
            )

    with pytest.raises(ValueError, match="finite"):
        _quality(NonFiniteModel(), teacher, evaluation)


def test_auto_requires_explicit_independent_nonempty_selection_data() -> None:
    model = _small_model()
    calibration = torch.randint(0, model.vocab, (4, model.block))
    with pytest.raises(ValueError, match="independent eval_data"):
        compress(model, method="auto", calibration_data=calibration, input_law=_input_law())
    with pytest.raises(ValueError, match="independent"):
        compress(
            model,
            method="auto",
            calibration_data=calibration,
            eval_data=calibration,
            input_law=_input_law(),
        )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("sample_budget", -1),
        ("sample_budget", 1.5),
        ("hybrid_sample_fraction", -0.1),
        ("hybrid_sample_fraction", np.nan),
        ("hybrid_sample_fraction", 1.1),
    ],
)
def test_compress_rejects_invalid_hybrid_sample_controls(name: str, value: object) -> None:
    model = _small_model()
    options = {name: value}
    with pytest.raises((TypeError, ValueError), match=name):
        compress(model, method="non_sampling", input_law=_input_law(), **options)


def test_hybrid_zero_cap_and_seeded_subset_never_exceed_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    teacher = _small_model()
    calibration = torch.arange(24, dtype=torch.long).reshape(6, 4) % teacher.vocab
    receipts = {"kept[0]<-block[0]": SimpleNamespace(surrogate_closure_error=1.0)}
    captured: list[torch.Tensor] = []

    def non_sampling(*args: object, **kwargs: object) -> object:
        return SimpleNamespace(model=copy.deepcopy(teacher), receipt_map=receipts)

    monkeypatch.setattr(compression, "_non_sampling", non_sampling)
    monkeypatch.setattr(
        compression,
        "_finetune_stages",
        lambda student, model, x, epochs, lr: captured.append(x.clone()),
    )

    _, _, zero = compression._hybrid(
        teacher,
        _input_law(),
        1.0,
        1.0,
        2,
        13,
        calibration,
        0.5,
        0,
        1,
        1,
        1.0e-3,
    )
    assert zero["sample_count"] == 0
    assert zero["sample_indices"] == []
    assert captured == []

    _, _, first = compression._hybrid(
        teacher,
        _input_law(),
        1.0,
        1.0,
        2,
        13,
        calibration,
        0.5,
        2,
        1,
        1,
        1.0e-3,
    )
    _, _, second = compression._hybrid(
        teacher,
        _input_law(),
        1.0,
        1.0,
        2,
        13,
        calibration,
        0.5,
        2,
        1,
        1,
        1.0e-3,
    )
    assert first["sample_count"] == 2
    assert first["sample_indices"] == second["sample_indices"]
    assert first["sample_indices"] != [0, 1]
    assert len(captured) == 2


def test_auto_pick_returns_receipts_for_selected_artifact_and_total_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    teacher = _small_model()
    ns_model, sk_model, hy_model = object(), object(), object()
    ns_receipts = {"ns": object()}
    hy_receipts = {"hy": object()}
    qualities = {id(ns_model): 0.2, id(sk_model): 0.4, id(hy_model): 0.9}

    monkeypatch.setattr(
        compression,
        "_non_sampling",
        lambda *args, **kwargs: SimpleNamespace(model=ns_model, receipt_map=ns_receipts),
    )
    monkeypatch.setattr(
        compression,
        "_sampling_kd",
        lambda *args, **kwargs: SimpleNamespace(student=sk_model),
    )
    monkeypatch.setattr(
        compression,
        "_hybrid",
        lambda *args, **kwargs: (
            hy_model,
            SimpleNamespace(receipt_map=hy_receipts),
            {"sample_count": 2, "selected_stages": ["hy"], "sample_indices": [1, 3]},
        ),
    )
    monkeypatch.setattr(compression, "_quality", lambda student, teacher, data: qualities[id(student)])

    method, artifact, candidates, extra = compression._auto_pick(
        teacher,
        torch.ones((4, 4), dtype=torch.long),
        torch.zeros((3, 4), dtype=torch.long),
        _input_law(),
        1.0,
        1.0,
        2,
        5,
        1,
        1.0e-3,
        0.5,
        2,
        1,
        1,
        1.0e-3,
        1,
    )
    assert method == "hybrid"
    assert artifact is hy_model
    assert extra["non_sampling_receipts"] is hy_receipts
    assert extra["hybrid_sample_indices"] == [1, 3]
    assert extra["selection_training_sample_count"] == 6
    assert extra["selection_evaluation_sample_count"] == 9
    assert all(candidate.evaluation_sample_count == 3 for candidate in candidates.values())


def test_compressed_auto_receipt_separates_deployment_and_selection_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _small_model()
    chosen = copy.deepcopy(model)
    candidates = {
        "non_sampling": MethodCandidate("non_sampling", 0.2, 0, 2, 0.2),
        "sampling_kd": MethodCandidate("sampling_kd", 0.4, 4, 2, 0.4),
        "hybrid": MethodCandidate("hybrid", 0.9, 1, 2, 0.9),
    }
    chosen_receipts = {"hy": object()}
    monkeypatch.setattr(
        compression,
        "_auto_pick",
        lambda *args, **kwargs: (
            "hybrid",
            chosen,
            candidates,
            {
                "non_sampling_receipts": chosen_receipts,
                "hybrid_selected_stages": ["hy"],
                "hybrid_sample_indices": [2],
                "selection_training_sample_count": 5,
                "selection_evaluation_sample_count": 6,
            },
        ),
    )
    calibration = torch.ones((4, 4), dtype=torch.long)
    evaluation = torch.zeros((2, 4), dtype=torch.long)
    result = compress(
        model,
        method="auto",
        calibration_data=calibration,
        eval_data=evaluation,
        input_law=_input_law(),
        kd_epochs=1,
        hybrid_epochs=1,
        n_mc=2,
    )

    assert result.model is chosen
    assert result.non_sampling_receipts is chosen_receipts
    assert result.receipt.sample_count == 1
    assert result.receipt.selection_training_sample_count == 5
    assert result.receipt.selection_evaluation_sample_count == 6
    assert result.receipt.total_selection_sample_count == 11
    assert result.hybrid_sample_indices == [2]


def test_fractional_token_ids_fail_before_any_compression_execution() -> None:
    model = _small_model()
    with pytest.raises(TypeError, match="integer token"):
        compress(
            model,
            method="sampling_kd",
            calibration_data=np.ones((2, 4), dtype=np.float64),
            input_law=_input_law(),
        )
