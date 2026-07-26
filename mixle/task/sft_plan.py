"""``sft_planner`` -- trace-SFT for generating tool plans with parser-gated validation.

The generative rung above :func:`~mixle.task.plan.distill_planner`. The step-students decompose by
classifying the next action; this trains a small causal LM (:class:`~mixle.models.LM`) on serialized teacher
traces with the prompt-masked SFT objective (``LM.fit_pairs``) so the whole plan is generated::

    request \\n=> tool(k=v; k=v) | tool(k=v) | done \\n

Free-form generation needs a strict validation boundary. The emitted text must parse under the plan grammar,
every tool must exist, and every required argument must be present. Anything else escalates to the teacher and
the trace is harvested. The model may emit arbitrary text, but only verified plans leave the function.

What this adds over the step-students (and what it does not): one model covers every tool with
variable-length plans and can generalize over entity values it did not see during training. It does not infer
unsupported tool compositions from absent traces; those cases escalate for teacher handling and future data
collection.
"""

from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mixle.task.toolcall import ToolSpec, _tool_spec_map, _validated_args, _validated_call

_EOS = "\n"
_PROMPT_SEP = "\n=> "
_EMPTY = "done"


def _encode_value(value: Any) -> str:
    """Encode one argument without colliding with the plan separators."""
    if isinstance(value, str) and value and not value.startswith("~") and not any(char in value for char in "();|=\n"):
        return value
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    except (TypeError, ValueError) as exc:
        raise ValueError("plan argument values must be finite JSON values") from exc
    encoded = base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
    return f"~{encoded}"


def _decode_value(token: str) -> Any:
    if not token.startswith("~"):
        return token
    encoded = token[1:]
    if not encoded:
        raise ValueError("empty escaped plan value")
    try:
        padded = encoded + "=" * (-len(encoded) % 4)
        value = json.loads(base64.b64decode(padded, altchars=b"-_", validate=True).decode())
    except (binascii.Error, ValueError, UnicodeDecodeError) as exc:
        raise ValueError("invalid escaped plan value") from exc
    if _encode_value(value) != token:
        raise ValueError("non-canonical escaped plan value")
    return value


def _serialize_plan(plan: Sequence[dict]) -> str:
    if not plan:
        return _EMPTY + _EOS
    parts = []
    for step in plan:
        args = "; ".join(f"{k}={_encode_value(v)}" for k, v in (step.get("args") or {}).items())
        parts.append(f"{step['tool']}({args})")
    return " | ".join(parts) + _EOS


def _parse_plan(text: str) -> list[dict] | None:
    """Strict inverse of :func:`_serialize_plan`; ``None`` when the text is not a well-formed plan."""
    body = text.split(_EOS, 1)[0].strip()
    if body == _EMPTY:
        return []
    steps: list[dict] = []
    for part in body.split(" | "):
        part = part.strip()
        if not (part and part.endswith(")") and "(" in part):
            return None
        name, arg_s = part[:-1].split("(", 1)
        if not name.isidentifier():
            return None
        args: dict[str, str] = {}
        if arg_s.strip():
            for kv in arg_s.split("; "):
                if "=" not in kv:
                    return None
                k, v = kv.split("=", 1)
                token = v.strip()
                if not k.strip().isidentifier() or not token or any(c in token for c in "()|=;") or k.strip() in args:
                    return None
                try:
                    args[k.strip()] = _decode_value(token)
                except ValueError:
                    return None
        steps.append({"tool": name, "args": args})
    return steps


def _validated_teacher_plan(plan: Any, specs: dict[str, ToolSpec]) -> list[dict[str, Any]]:
    if isinstance(plan, (str, bytes)) or not isinstance(plan, Sequence):
        raise ValueError("teacher plan must be a sequence of tool-call dictionaries")
    validated: list[dict[str, Any]] = []
    for step in plan:
        call = _validated_call(step, specs)
        if call["tool"] is None:
            raise ValueError("teacher plans must represent no action as an empty plan")
        validated.append(call)
    return validated


class _CharCodec:
    """A compact char-level codec (pad=0, unk=1) built from the training corpus."""

    def __init__(self, corpus: Sequence[str]) -> None:
        chars = sorted(set("".join(corpus)) | {_EOS})
        self.itos = ["\x00", "\x01", *chars]
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.eos_id = self.stoi[_EOS]

    @classmethod
    def from_itos(cls, itos: Sequence[str]) -> _CharCodec:
        codec = cls.__new__(cls)
        codec.itos = list(itos)
        codec.stoi = {c: i for i, c in enumerate(codec.itos)}
        codec.eos_id = codec.stoi[_EOS]
        return codec

    @property
    def vocab(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(c, 1) for c in text]

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(self.itos[i] if 0 <= int(i) < len(self.itos) else "" for i in ids)


@dataclass
class GenerativePlanner:
    """A plan-writing LM behind a parse-and-validate gate: only verified plans leave; the rest escalate."""

    lm: Any
    codec: _CharCodec
    tools: dict[str, ToolSpec]
    teacher: Callable[[str], list[dict]]
    plan_agreement: float
    calibration_agreement: float | None = None
    calibration_size: int = 0
    test_size: int = 0
    max_new: int = 160
    constrained: bool = True  # decode inside the plan grammar (invalid output unrepresentable)
    conf_floor: float | None = None  # calibrated mean-logprob floor: low-confidence decodes escalate
    lm_config: dict = field(default_factory=dict)  # builder config for save/load round-trip
    n_requests: int = 0
    n_escalated: int = 0
    harvested: list[tuple[str, list[dict]]] = field(default_factory=list)

    def _validate(self, plan: list[dict] | None, request: str) -> bool:
        if plan is None:
            return False
        for step in plan:
            spec = self.tools.get(step["tool"])
            if spec is None:
                return False
            try:
                args = _validated_args(spec, step["args"])
            except ValueError:
                return False
            # copy-fidelity: plan arguments are EXTRACTIVE — a generated value that does not literally
            # occur in the request (or the tool's own fixed vocabulary, e.g. kind=refund) is a silent
            # copy error (order 4242 -> order_id=4202) that spec validity cannot catch. Reject it.
            for argument, value in args.items():
                copied = bool(str(value)) and str(value) in request
                declared = any(
                    _encode_value(value) == _encode_value(candidate) for candidate in spec.fixed_values(argument)
                )
                if not copied and not declared:
                    return False
        return True

    def try_plan(self, request: str) -> list[dict] | None:
        """Generate, parse, validate (grammar + specs + copy-fidelity); ``None`` = must escalate.

        With ``constrained=True`` (default) the decode itself runs inside the plan grammar
        (:func:`mixle.task.constrained.constrained_plan_decode`): malformed text and copy-drifted
        values are unrepresentable, and the parse/validate below is a pure backstop."""
        request = str(request)
        if self.constrained:
            from mixle.task.constrained import constrained_plan_decode

            decoded = constrained_plan_decode(self.lm, self.codec, request, self.tools, max_new=self.max_new)
            if decoded is None:
                return None
            text, conf = decoded
            # the grammar guarantees form, not content: a weak model can write a well-formed wrong plan,
            # so emission additionally requires the model's own confidence to clear the calibrated floor
            if self.conf_floor is not None and conf < self.conf_floor:
                return None
        else:
            prompt = self.codec.encode(request + _PROMPT_SEP)
            out = self.lm.generate(prompt, n=self.max_new, greedy=True, stop_id=self.codec.eos_id)
            text = self.codec.decode(out[len(prompt) :])
        plan = _parse_plan(text if text.endswith(_EOS) else text + "")
        return plan if self._validate(plan, request) else None

    def __call__(self, request: str) -> dict[str, Any]:
        self.n_requests += 1
        plan = self.try_plan(request)
        if plan is not None:
            return {"plan": plan, "escalate": False}
        self.n_escalated += 1
        want = _validated_teacher_plan(self.teacher(request), self.tools)
        self.harvested.append((request, want))
        return {"plan": [dict(p) for p in want], "escalate": True}

    def report(self) -> dict[str, Any]:
        """Return plan agreement, escalation, and harvested-trace metrics."""
        return {
            "plan_agreement": round(self.plan_agreement, 4),
            "calibration_agreement": (
                round(self.calibration_agreement, 4) if self.calibration_agreement is not None else None
            ),
            "calibration_size": self.calibration_size,
            "test_size": self.test_size,
            "requests": self.n_requests,
            "escalated": self.n_escalated,
            "escalation_rate": (self.n_escalated / self.n_requests) if self.n_requests else 0.0,
            "harvested_traces": len(self.harvested),
        }

    def save(self, path: str) -> str:
        """Persist the plan-writing LM (weights + builder config), codec, specs, and gates; :meth:`load` restores."""
        import json
        from pathlib import Path

        from mixle.task.artifact import save_module

        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        save_module(str(out / "lm"), self.lm.module, "mixle.causal_lm", dict(self.lm_config), task="plan-generation")
        manifest = {
            "kind": "genplanner/v1",
            "itos": self.codec.itos,
            "tools": {
                n: {
                    "args": t.args,
                    "required": t.required,
                    "vocabulary": t.vocabulary,
                }
                for n, t in self.tools.items()
            },
            "plan_agreement": self.plan_agreement,
            "calibration_agreement": self.calibration_agreement,
            "calibration_size": self.calibration_size,
            "test_size": self.test_size,
            "max_new": self.max_new,
            "constrained": self.constrained,
            "conf_floor": self.conf_floor,
            "lm_config": dict(self.lm_config),
        }
        (out / "genplanner.json").write_text(json.dumps(manifest, indent=2))
        return str(out)

    @classmethod
    def load(cls, path: str, teacher: Callable[[str], list[dict]], *, device: str = "cpu") -> GenerativePlanner:
        """Reconstitute a serving GenerativePlanner from :meth:`save` output plus the teacher fallback."""
        import json
        from pathlib import Path

        from mixle.models import LM
        from mixle.task.artifact import load_module

        p = Path(path)
        manifest = json.loads((p / "genplanner.json").read_text())
        cfg = dict(manifest["lm_config"])
        lm = LM(device=device, **cfg)
        module, _ = load_module(str(p / "lm"), device=device)
        lm.module = module
        tools = _tool_spec_map(
            [
                ToolSpec(
                    n,
                    list(t["args"]),
                    t.get("required"),
                    t.get("vocabulary"),
                )
                for n, t in manifest["tools"].items()
            ],
            reserved=(_EMPTY,),
        )
        return cls(
            lm=lm,
            codec=_CharCodec.from_itos(manifest["itos"]),
            tools=tools,
            teacher=teacher,
            plan_agreement=float(manifest.get("plan_agreement", float("nan"))),
            calibration_agreement=manifest.get("calibration_agreement"),
            calibration_size=int(manifest.get("calibration_size", 0)),
            test_size=int(manifest.get("test_size", 0)),
            max_new=int(manifest.get("max_new", 160)),
            constrained=bool(manifest.get("constrained", True)),
            conf_floor=manifest.get("conf_floor"),
            lm_config=cfg,
        )


def _plans_match(got: list[dict], want: list[dict], specs: dict[str, ToolSpec]) -> bool:
    try:
        got_validated = _validated_teacher_plan(got, specs)
        want_validated = _validated_teacher_plan(want, specs)
    except (TypeError, ValueError):
        return False
    if len(got_validated) != len(want_validated):
        return False
    for g, w in zip(got_validated, want_validated):
        if g["tool"] != w["tool"]:
            return False
        if set(g["args"]) != set(w["args"]):
            return False
        if any(_encode_value(g["args"][argument]) != _encode_value(w["args"][argument]) for argument in g["args"]):
            return False
    return True


def sft_planner(
    teacher: Callable[[str], list[dict]],
    requests: Sequence[str],
    tools: Sequence[ToolSpec],
    *,
    holdout: float = 0.2,
    seed: int = 0,
    d_model: int = 96,
    n_layer: int = 3,
    n_head: int = 4,
    block: int = 192,
    epochs: int = 30,
    lr: float = 3e-3,
    device: str = "cpu",
    constrained: bool = True,
) -> GenerativePlanner:
    """Trace-SFT a small causal LM into a plan writer, verified on held-out requests.

    Traces serialize as ``request\\n=> tool(k=v; ...) | ... \\n`` pairs; ``LM.fit_pairs`` trains with the
    prompt masked so only plan tokens carry loss; generation stops at newline. Held-out agreement is
    plan-level exact match (tools + required args, in order) on requests the LM never saw.
    """
    import torch

    from mixle.models import LM

    if not callable(teacher):
        raise TypeError("teacher must be callable")
    if isinstance(requests, (str, bytes)):
        raise TypeError("requests must be a sequence of request strings")
    reqs = list(requests)
    if any(not isinstance(request, str) or not request for request in reqs):
        raise ValueError("requests must contain non-empty strings")
    if len(reqs) < 16:
        raise ValueError("sft_planner needs at least 16 example requests")
    if (
        isinstance(holdout, (bool, np.bool_))
        or not isinstance(holdout, (int, float, np.integer, np.floating))
        or not np.isfinite(holdout)
        or not 0.0 < float(holdout) < 1.0
    ):
        raise ValueError("holdout must be a finite fraction strictly between 0 and 1")
    specs = _tool_spec_map(tools, reserved=(_EMPTY,))
    # Validate and bind every teacher trace exactly once. Duplicate requests
    # remain distinct rows and cannot collapse through a request-keyed dict.
    teacher_plans = [_validated_teacher_plan(teacher(request), specs) for request in reqs]

    torch.manual_seed(seed)  # LM weight init draws from torch's global RNG; pin it so seed= means seed
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(reqs))
    n_evaluation = min(
        len(reqs) - 4,
        max(4, int(round(len(reqs) * float(holdout)))),
    )
    n_calibration = n_evaluation // 2
    calibration_indices = [int(index) for index in order[:n_calibration]]
    test_indices = [int(index) for index in order[n_calibration:n_evaluation]]
    train_indices = [int(index) for index in order[n_evaluation:]]

    prompts = [reqs[index] + _PROMPT_SEP for index in train_indices]
    completions = [_serialize_plan(teacher_plans[index]) for index in train_indices]
    codec = _CharCodec([*prompts, *completions])
    pairs = [(codec.encode(prompt), codec.encode(completion)) for prompt, completion in zip(prompts, completions)]

    lm = LM(vocab=codec.vocab, d_model=d_model, n_layer=n_layer, n_head=n_head, block=block, device=device)
    lm.fit_pairs(pairs, epochs=epochs, lr=lr, seed=seed)

    planner = GenerativePlanner(
        lm=lm,
        codec=codec,
        tools=specs,
        teacher=teacher,
        plan_agreement=float("nan"),
        calibration_size=len(calibration_indices),
        test_size=len(test_indices),
        max_new=block,
        constrained=constrained,
        lm_config={"vocab": codec.vocab, "d_model": d_model, "n_layer": n_layer, "n_head": n_head, "block": block},
    )
    if constrained:
        # calibrate the confidence floor on the holdout: wrong-but-well-formed decodes score lower than
        # correct ones, so pick the floor that keeps (almost) all correct decodes and pushes above the
        # wrong ones when possible — low-confidence generations then escalate instead of shipping
        from mixle.task.constrained import constrained_plan_decode

        correct_scores: list[float] = []
        wrong_scores: list[float] = []
        for index in calibration_indices:
            request = reqs[index]
            decoded = constrained_plan_decode(lm, codec, request, specs, max_new=block)
            if decoded is None:
                continue
            plan = _parse_plan(decoded[0] if decoded[0].endswith(_EOS) else decoded[0] + _EOS)
            ok = plan is not None and _plans_match(plan, teacher_plans[index], specs)
            (correct_scores if ok else wrong_scores).append(decoded[1])
        if correct_scores:
            floor = float(np.quantile(correct_scores, 0.05))
            if wrong_scores:
                floor = max(floor, min(float(max(wrong_scores)) + 1e-9, float(np.quantile(correct_scores, 0.5))))
            planner.conf_floor = floor
    calibration_agree = 0
    for index in calibration_indices:
        got = planner.try_plan(reqs[index])
        calibration_agree += int(got is not None and _plans_match(got, teacher_plans[index], specs))
    planner.calibration_agreement = calibration_agree / len(calibration_indices)

    test_agree = 0
    for index in test_indices:
        got = planner.try_plan(reqs[index])
        test_agree += int(got is not None and _plans_match(got, teacher_plans[index], specs))
    planner.plan_agreement = test_agree / len(test_indices)
    return planner


def score_plan(planner: GenerativePlanner, request: str, plan: Sequence[dict]) -> float:
    """Mean per-character teacher-forced log-probability of a candidate ``plan`` under the trained LM.

    This is not a decode: it scores a plan supplied by the CALLER (a candidate to rank against
    alternatives, or an already-taken plan to flag as low-probability after the fact) -- the same
    confidence metric :func:`~mixle.task.constrained.constrained_plan_decode` computes for its own
    greedy path, generalized to any plan text. Higher (less negative) is more probable; a plan scoring
    below the planner's calibrated ``conf_floor`` is exactly the "low-probability plan" escalation
    signal used by plan-quality checks, computed explicitly here rather than left implicit in the
    decode loop.
    """
    import torch

    text = _serialize_plan(list(plan))
    lm = planner.lm
    w = planner.codec.encode(str(request) + _PROMPT_SEP)
    ids = planner.codec.encode(text)
    logps: list[float] = []
    was_training = bool(lm.module.training)
    lm.module.to(lm.device).eval()
    try:
        with torch.no_grad():
            for ch_id in ids:
                win = w[-lm.block :]
                logits = lm.module(torch.as_tensor([win], dtype=torch.float32).to(lm.device))[0].cpu().numpy()
                lse = float(np.logaddexp.reduce(logits - logits.max()) + logits.max())
                logps.append(float(logits[ch_id]) - lse)
                w.append(ch_id)
    finally:
        lm.module.train(was_training)
    return float(np.mean(logps)) if logps else float("-inf")


def sample_plans(
    planner: GenerativePlanner, request: str, n: int = 5, *, temperature: float = 1.0, seed: int = 0
) -> list[tuple[list[dict] | None, float]]:
    """Draw ``n`` stochastic candidate plans from the trained LM, each scored by :func:`score_plan`.

    Sorted highest-score first. A draw that fails to parse or validate (the grammar is not enforced
    during stochastic sampling, unlike the constrained decode path) is returned as ``(None, -inf)`` --
    an undefined score IS the escalation signal: a generative decomposition model that cannot produce a
    coherent plan for a request should say so, never guess silently.
    """
    prompt = planner.codec.encode(str(request) + _PROMPT_SEP)
    out: list[tuple[list[dict] | None, float]] = []
    for i in range(int(n)):
        gen = planner.lm.generate(
            prompt,
            n=planner.max_new,
            temperature=temperature,
            greedy=False,
            seed=seed + i,
            stop_id=planner.codec.eos_id,
        )
        text = planner.codec.decode(gen[len(prompt) :])
        plan = _parse_plan(text if text.endswith(_EOS) else text + _EOS)
        if plan is not None and planner._validate(plan, request):
            out.append((plan, score_plan(planner, request, plan)))
        else:
            out.append((None, float("-inf")))
    return sorted(out, key=lambda pair: pair[1], reverse=True)
