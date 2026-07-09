"""Secret detection and redaction for substrate text.

Knowledge flows into the substrate from documents, traces, and tool outputs -- exactly the places a
leaked credential hides. :func:`detect_secrets` scans text for well-known secret shapes (API keys,
bearer tokens, AWS keys, private-key blocks, credentials embedded in URLs, and ``key=value``
assignments of sensitive names); :func:`redact_secrets` masks them in place; :func:`scan_item` /
:func:`scan_substrate` sweep stored items and :func:`safe_text` gives a redact-before-store guard.

    The patterns are deliberately conservative and named -- each finding says which rule matched, so a
false positive is inspectable rather than mysterious. This is detection, not a vault: it catches the
common leaks (a pasted key, a token in a log line) so they don't get indexed and served, and it flags
the rest for review. Redaction preserves a short prefix so a human can still recognize which key it was
without exposing the secret.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# each rule: (name, compiled pattern). Ordered most-specific first so overlapping matches attribute well.
_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("private_key_block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S)),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")),
    ("url_credentials", re.compile(r"\b[a-z][a-z0-9+.\-]*://[^\s:/@]+:([^\s:/@]+)@")),
    ("bearer_token", re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{16,}\b")),
    # key=value / key: value assignments of sensitive names with a non-trivial value
    (
        "sensitive_assignment",
        re.compile(
            r"(?i)\b(?:api[_-]?key|secret|token|passwd|password|access[_-]?key|private[_-]?key)\b"
            r"\s*[:=]\s*['\"]?([^\s'\"]{6,})"
        ),
    ),
]


@dataclass
class SecretFinding:
    """One detected secret: which rule matched, where, and a safe preview (the value stays masked)."""

    rule: str
    start: int
    end: int
    preview: str  # a short, non-sensitive hint (rule + first few chars)


@dataclass
class SecretScan:
    """The result of scanning a text: whether anything leaked and every finding."""

    findings: list[SecretFinding] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        """Whether the scan found no secrets."""
        return not self.findings

    def rules(self) -> list[str]:
        """Return the sorted names of triggered secret-detection rules."""
        return sorted({f.rule for f in self.findings})

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable scan summary."""
        return {
            "clean": self.clean,
            "n_findings": len(self.findings),
            "rules": self.rules(),
            "findings": [{"rule": f.rule, "start": f.start, "end": f.end, "preview": f.preview} for f in self.findings],
        }


def _preview(rule: str, matched: str) -> str:
    head = matched[:4]
    return f"{rule}:{head}…" if len(matched) > 4 else f"{rule}:{matched}"


def detect_secrets(text: str) -> SecretScan:
    """Scan ``text`` for well-known secret shapes; return a :class:`SecretScan` naming each finding."""
    if not text:
        return SecretScan()
    findings: list[SecretFinding] = []
    claimed: list[tuple[int, int]] = []  # spans already attributed, so a more-specific rule wins
    for rule, pattern in _RULES:
        for m in pattern.finditer(text):
            span = m.span()
            if any(span[0] < c1 and span[1] > c0 for c0, c1 in claimed):
                continue  # overlaps an already-claimed (more specific) finding
            claimed.append(span)
            findings.append(SecretFinding(rule=rule, start=span[0], end=span[1], preview=_preview(rule, m.group(0))))
    findings.sort(key=lambda f: f.start)
    return SecretScan(findings=findings)


def redact_secrets(text: str, *, mask: str = "[REDACTED:{rule}]", keep_prefix: int = 0) -> str:
    """Return ``text`` with every detected secret replaced by a rule-labelled mask (destructive to secrets).

    ``keep_prefix`` leaves that many leading characters of the secret visible (0 = fully masked) so a
    reader can still tell which credential it was without recovering it."""
    scan = detect_secrets(text)
    if scan.clean:
        return text
    out = []
    cursor = 0
    for f in scan.findings:
        out.append(text[cursor : f.start])
        secret = text[f.start : f.end]
        prefix = secret[:keep_prefix] if keep_prefix > 0 else ""
        out.append(prefix + mask.format(rule=f.rule))
        cursor = f.end
    out.append(text[cursor:])
    return "".join(out)


def safe_text(text: str) -> str:
    """Redact-before-store guard: mask any secrets so they are never indexed or served."""
    return redact_secrets(text)


def scan_item(item: Any) -> SecretScan:
    """Scan a substrate item's text surface for secrets."""
    return detect_secrets(getattr(item, "text", "") or "")


def scan_substrate(substrate: Any, *, scope: str | None = None) -> dict[str, Any]:
    """Sweep a substrate for leaked secrets and report which stored items triggered rules.

    Returns ``{n_items, n_dirty, dirty: [{item_id, rules}]}`` for compatibility with existing callers; entries in
    ``dirty`` are the items that matched one or more secret-detection rules."""
    items = substrate.all(scope=scope)
    dirty: list[dict[str, Any]] = []
    for it in items:
        scan = scan_item(it)
        if not scan.clean:
            dirty.append({"item_id": it.id, "rules": scan.rules()})
    return {"n_items": len(items), "n_dirty": len(dirty), "dirty": dirty}
