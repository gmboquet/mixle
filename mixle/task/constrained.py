"""Grammar-constrained plan decoding -- invalid output becomes impossible, not just detected.

:mod:`~mixle.task.sft_plan` generates freely and then gates (parse + specs + copy-fidelity). This module
moves the gate INTO the decoder: a character-level automaton over the plan grammar masks the LM's logits
at every step, so the model can only ever choose among characters that keep the output a valid plan --
tool names come from the specs, argument keys from the chosen tool, and argument VALUES are anchored to
the request text (each value character must extend a live substring match), which makes the silent copy
error (``order 4242 -> order_id=4202``) unrepresentable: after ``42`` the only continuation the request
offers is ``4``.

The practical payoff is largest for small or undertrained models: instead of having to place all its
probability mass on exactly the right free-form string, the LM only ranks the handful of legal
continuations -- so constrained decoding lifts agreement precisely where compute is scarce.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from mixle.task.toolcall import ToolSpec

_EOS = "\n"
_DONE = "done"


@dataclass(frozen=True)
class _State:
    mode: str  # "name" | "key" | "value" | "step_end" | "terminal"
    prefix: str = ""
    tool: str | None = None
    used: frozenset = frozenset()
    val_pos: tuple = ()  # live substring-match start positions in the request
    val_len: int = 0
    pending: str = ""  # forced literal still to emit
    first_step: bool = True


class PlanGrammar:
    """The character automaton: ``allowed(state)`` and ``advance(state, char)`` over the plan DSL."""

    def __init__(self, specs: dict[str, ToolSpec], request: str) -> None:
        self.specs = specs
        self.request = str(request)

    def start(self) -> _State:
        """Return the initial parser state."""
        return _State(mode="name")

    def _name_candidates(self, s: _State) -> list[str]:
        names = sorted(self.specs)
        return [*names, _DONE] if s.first_step else names

    def allowed(self, s: _State) -> set[str]:
        """Return characters allowed from parser state ``s``."""
        if s.pending:
            return {s.pending[0]}
        if s.mode == "terminal":
            return set()
        if s.mode == "name":
            out: set[str] = set()
            for cand in self._name_candidates(s):
                if cand.startswith(s.prefix) and len(cand) > len(s.prefix):
                    out.add(cand[len(s.prefix)])
            if s.prefix in self.specs:
                out.add("(")
            if s.first_step and s.prefix == _DONE:
                out.add(_EOS)
            return out
        if s.mode == "key":
            spec = self.specs[s.tool or ""]
            cands = [a for a in spec.args if a not in s.used]
            out = set()
            for cand in cands:
                if cand.startswith(s.prefix) and len(cand) > len(s.prefix):
                    out.add(cand[len(s.prefix)])
            if s.prefix in cands:
                out.add("=")
            return out
        if s.mode == "value":
            if s.val_len == 0:
                cont = set(self.request)
            else:
                cont = {self.request[p + s.val_len] for p in s.val_pos if p + s.val_len < len(self.request)}
            cont -= set("()|=;\n")  # the parser bans structural characters inside values
            out = set(cont)
            if s.val_len > 0:  # a non-empty value may terminate: ")" ends the step, "; " starts the next key
                out.add(")")
                if any(a not in s.used for a in self.specs[s.tool or ""].args):
                    out.add(";")
            return out
        if s.mode == "step_end":
            return {" ", _EOS}
        raise AssertionError(f"unknown mode {s.mode!r}")

    def advance(self, s: _State, c: str) -> _State:
        """Advance parser state ``s`` by one character."""
        if s.pending:
            rest = s.pending[1:]
            if rest:
                return replace(s, pending=rest)
            # a consumed literal lands where its mode said it would
            return replace(s, pending="")
        if s.mode == "name":
            if c == "(":
                spec = self.specs[s.prefix]
                if spec.args:
                    return replace(s, mode="key", tool=s.prefix, prefix="", used=frozenset())
                return replace(s, mode="step_end", tool=s.prefix, prefix="", pending=")")
            if c == _EOS:
                return replace(s, mode="terminal")
            return replace(s, prefix=s.prefix + c)
        if s.mode == "key":
            if c == "=":
                return replace(s, mode="value", used=s.used | {s.prefix}, prefix="", val_pos=(), val_len=0)
            return replace(s, prefix=s.prefix + c)
        if s.mode == "value":
            if c == ")":
                return replace(s, mode="step_end", val_pos=(), val_len=0)
            if c == ";":
                return replace(s, mode="key", prefix="", val_pos=(), val_len=0, pending=" ")
            if s.val_len == 0:
                pos = tuple(i for i, ch in enumerate(self.request) if ch == c)
            else:
                pos = tuple(
                    p for p in s.val_pos if p + s.val_len < len(self.request) and self.request[p + s.val_len] == c
                )
            return replace(s, val_pos=pos, val_len=s.val_len + 1)
        if s.mode == "step_end":
            if c == _EOS:
                return replace(s, mode="terminal")
            return replace(s, mode="name", prefix="", tool=None, first_step=False, pending="| ")
        raise AssertionError(f"cannot advance mode {s.mode!r}")


def constrained_plan_decode(
    lm: Any, codec: Any, request: str, specs: dict[str, ToolSpec], *, max_new: int = 200
) -> tuple[str, float] | None:
    """Greedy decode with the grammar mask: the highest-logit LEGAL character at every step.

    Returns ``(plan_text, confidence)`` — the confidence is the mean full-vocabulary log-probability
    the model itself put on the legal path it took — or ``None`` when the automaton dead-ends. The
    grammar guarantees FORM; a well-formed wrong plan is still possible from a weak model, so callers
    gate emission on a calibrated confidence floor (see ``sft_planner``).
    """
    import torch

    from mixle.task.sft_plan import _PROMPT_SEP

    grammar = PlanGrammar(specs, request)
    state = grammar.start()
    w = codec.encode(str(request) + _PROMPT_SEP)
    lm.module.to(lm.device).eval()
    text: list[str] = []
    logps: list[float] = []
    try:
        for _ in range(int(max_new)):
            allowed = grammar.allowed(state)
            if not allowed:
                return None
            ids = [codec.stoi[c] for c in allowed if c in codec.stoi]
            if not ids:
                return None
            win = w[-lm.block :]
            with torch.no_grad():
                logits = lm.module(torch.as_tensor([win], dtype=torch.float32).to(lm.device))[0].cpu().numpy()
            masked = np.full_like(logits, -np.inf)
            masked[ids] = logits[ids]
            nxt = int(masked.argmax())
            lse = float(np.logaddexp.reduce(logits - logits.max()) + logits.max())
            logps.append(float(logits[nxt]) - lse)
            ch = codec.itos[nxt]
            w.append(nxt)
            text.append(ch)
            state = grammar.advance(state, ch)
            if state.mode == "terminal":
                return "".join(text), float(np.mean(logps))
        return None  # budget exhausted before EOS
    finally:
        lm.module.train()
