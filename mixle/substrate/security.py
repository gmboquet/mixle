"""Secret detection and redaction for substrate text and structured data.

Knowledge flows into the substrate from documents, traces, and tool outputs -- exactly the places a
leaked credential hides, and not only as free text: a payload dict (a harvested trace, a manifest, a
record) carries one just as easily, and lexical retrieval serializes ``payload`` + ``tags`` into the
same searchable surface as ``.text`` (see ``mixle.substrate.core._lexical_score``). :func:`detect_secrets`
scans a string for well-known secret shapes (API keys, bearer tokens, AWS keys, private-key blocks,
credentials embedded in URLs, and ``key=value`` assignments of sensitive names); :func:`redact_secrets`
masks them in place; :func:`item_surface` builds the same text+payload+tags surface lexical search
indexes, so :func:`scan_item` / :func:`scan_substrate` sweep everything reachable through retrieval, not
just ``.text``; :func:`redact_value` is :func:`redact_secrets` recursed over a whole JSON-like structure;
:func:`safe_text` is the redact-before-store guard for free text, and :func:`enforce_secret_policy` is
the same guard for a whole item -- the store-boundary choke point ``Substrate.put()`` / ``.update()``
route every write through, so the guard is mandatory rather than something a caller opts into.

The patterns are deliberately conservative and named -- each finding says which rule matched, so a
false positive is inspectable rather than mysterious. This is detection, not a vault: it catches the
common leaks (a pasted key, a token in a log line) so they don't get indexed and served, and it flags
the rest for review. Redaction preserves a short prefix so a human can still recognize which key it was
without exposing the secret.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from typing import Any, Literal

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


# Redaction is applied until it stops changing the text. A rule can match only the credential and
# leave its assignment context, so the mask itself may re-trigger a broader rule; see redact_secrets.
_MAX_REDACTION_PASSES = 8


def _redact_once(text: str, *, mask: str, keep_prefix: int) -> str:
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


def redact_secrets(text: str, *, mask: str = "[REDACTED:{rule}]", keep_prefix: int = 0) -> str:
    """Return ``text`` with every detected secret replaced by a rule-labelled mask (destructive to secrets).

    ``keep_prefix`` leaves that many leading characters of the secret visible (0 = fully masked) so a
    reader can still tell which credential it was without recovering it.

    Applied to a FIXED POINT, because one pass did not deliver what this docstring promises
    (MXR-080-1882). A rule may match only the credential and leave its assignment context in place:
    ``token=ghp_...`` masks to ``token=[REDACTED:github_token]``, which still matches the
    ``sensitive_assignment`` rule, so the "redacted" text was still detected as carrying a secret. That
    only became visible once the store began re-scanning what it was about to write; before that, the
    single pass simply returned text nobody checked again.

    Iteration is bounded and monotone in practice -- each pass replaces at least one match with a mask
    that is not itself that rule's shape -- but the bound is explicit rather than assumed, and a text
    that has not converged is refused instead of returned dirty.
    """
    current = text
    for _ in range(_MAX_REDACTION_PASSES):
        nxt = _redact_once(current, mask=mask, keep_prefix=keep_prefix)
        if nxt == current:
            return current
        current = nxt
    if not detect_secrets(current).clean:
        raise ValueError(
            f"redaction did not converge in {_MAX_REDACTION_PASSES} passes; the masked text still "
            "matches a secret rule. Refusing to return text that would be stored in the clear."
        )
    return current


def safe_text(text: str) -> str:
    """Redact-before-store guard: mask any secrets so they are never indexed or served."""
    return redact_secrets(text)


def item_surface(item: Any) -> str:
    """The full text a substrate item exposes to indexed/serialized surfaces: text + payload + tags.

    Mirrors ``mixle.substrate.core._lexical_score``'s surface construction exactly (same join, same
    ``json.dumps(payload)``), so this scans precisely what lexical retrieval makes reachable -- a
    secret embedded anywhere in ``payload`` (however deeply nested; ``json.dumps`` flattens it into the
    same string lexical search tokenizes) or in ``tags`` is caught, not just a secret sitting in
    ``.text``. Falls back to ``str(payload)`` when ``payload`` is not JSON-serializable, so a scan can
    never raise on data that would otherwise be storable (search's own lexical path would raise on that
    same payload lazily, at query time; scanning must not be the thing that raises earlier at write time).
    """
    text = getattr(item, "text", "") or ""
    payload = getattr(item, "payload", None) or {}
    tags = getattr(item, "tags", None) or []
    try:
        payload_json = json.dumps(payload)
    except (TypeError, ValueError):
        payload_json = str(payload)
    return " ".join([text, payload_json, " ".join(str(t) for t in tags)])


def redact_value(value: Any, *, mask: str = "[REDACTED:{rule}]", keep_prefix: int = 0) -> Any:
    """Recursively redact secrets from a stored value, over the whole surface :func:`item_surface` reads.

    Redaction has to cover exactly what the scanner covers, and it did not (MXR-080-1882).
    ``item_surface`` serializes ``payload`` with ``json.dumps`` -- which writes dictionary KEYS as well
    as values -- and falls back to ``str(payload)`` when that fails, which stringifies sets and any
    opaque object's ``repr``. This function reached only dict values, lists and tuples, so three
    reachable places were detected by the scan and then stored in the clear:

    * a dictionary key, e.g. ``{"api_key=abcdefghi": "v"}``. Detected, stored intact.
    * a ``set``/``frozenset``, which fell to the untouched-passthrough at the bottom.
    * an opaque object whose ``str``/``repr`` carries the secret, likewise.

    Keys are redacted too, and a redaction that would collapse two distinct keys onto one masked
    string raises rather than silently dropping an entry -- losing a payload field to sanitization is
    a different failure from masking one. An opaque leaf is replaced by its redacted string form only
    when its own string carries a secret; otherwise it passes through unchanged, since converting
    every opaque value to text would destroy payloads that were never at risk.
    """
    if isinstance(value, str):
        return redact_secrets(value, mask=mask, keep_prefix=keep_prefix)
    if isinstance(value, dict):
        redacted: dict[Any, Any] = {}
        for key, item in value.items():
            safe_key = redact_secrets(key, mask=mask, keep_prefix=keep_prefix) if isinstance(key, str) else key
            if safe_key in redacted:
                raise ValueError(
                    f"redacting payload keys collapsed two distinct keys onto {safe_key!r}; storing "
                    "the sanitized payload would silently drop a field. Rename the key that carries "
                    "the secret, or store it under secret_policy='reject'."
                )
            redacted[safe_key] = redact_value(item, mask=mask, keep_prefix=keep_prefix)
        return redacted
    if isinstance(value, list):
        return [redact_value(v, mask=mask, keep_prefix=keep_prefix) for v in value]
    if isinstance(value, tuple):
        return tuple(redact_value(v, mask=mask, keep_prefix=keep_prefix) for v in value)
    if isinstance(value, (set, frozenset)):
        # Reachable through item_surface's str(payload) fallback, and previously untouched.
        redacted_members = {redact_value(v, mask=mask, keep_prefix=keep_prefix) for v in value}
        return frozenset(redacted_members) if isinstance(value, frozenset) else redacted_members
    if value is None or isinstance(value, (bool, int, float)):
        return value
    rendered = str(value)
    if detect_secrets(rendered).clean:
        return value
    # The object's own text carries a secret and that text is what reaches the stored/indexed
    # surface, so the masked text is what may be kept.
    return redact_secrets(rendered, mask=mask, keep_prefix=keep_prefix)


def scan_item(item: Any) -> SecretScan:
    """Scan a substrate item's full indexed/serialized surface (text + payload + tags) for secrets."""
    return detect_secrets(item_surface(item))


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


# The two supported store-boundary behaviors when a write's surface (item_surface) turns up a secret.
# "redact" masks in place and stores the sanitized result; "reject" stores nothing and raises instead.
SecretPolicy = Literal["redact", "reject"]
SECRET_POLICIES: tuple[SecretPolicy, ...] = ("redact", "reject")


class SecretPolicyError(ValueError):
    """Raised when ``secret_policy="reject"`` rejects a write because a secret was detected.

    Nothing is written when this is raised -- the caller's item never reaches the store."""

    def __init__(self, scan: SecretScan, *, item_id: str | None = None) -> None:
        self.scan = scan
        self.item_id = item_id
        where = f" (item {item_id!r})" if item_id else ""
        super().__init__(
            f"secret_policy='reject': detected {len(scan.findings)} secret(s){where}, "
            f"rules={scan.rules()}; nothing was written"
        )


def enforce_secret_policy(item: Any, *, policy: SecretPolicy = "redact") -> tuple[Any, SecretScan]:
    """The store-boundary secret guard: scan every indexed/serialized field of ``item`` -- text,
    payload (recursively), and tags, i.e. exactly :func:`item_surface` -- and act on what's found.

    This is the single choke point ``Substrate.put()`` / ``.update()`` route every write through, so
    the redact-before-store guard is no longer optional: previously :func:`safe_text` existed but
    nothing called it, and scanning (:func:`scan_item` / :func:`scan_substrate`) only ever looked at
    ``.text``, silently missing a secret embedded in ``payload`` (reachable through lexical retrieval's
    own ``json.dumps(payload)`` serialization -- see ``mixle.substrate.core._lexical_score``) or in
    ``tags``.

    Args:
        item: a substrate item -- read via ``.text`` / ``.payload`` / ``.tags`` and, when a secret is
            found under ``policy="redact"``, rebuilt with :func:`dataclasses.replace`, so any dataclass
            with those three fields works, not only :class:`~mixle.substrate.core.SubstrateItem`.
        policy: ``"redact"`` (default) returns a sanitized copy with every detected secret masked in
            place (text via :func:`safe_text`, payload via :func:`redact_value`, tags via
            :func:`safe_text` per entry) -- so a secret can be written, embedded, and served back only
            in masked form, never in the clear. ``"reject"`` raises :class:`SecretPolicyError` instead
            of returning anything -- nothing is written.

    Returns:
        ``(item_to_store, scan)`` where ``scan`` is the *pre-redaction* :class:`SecretScan` (what was
        found before any masking, empty when nothing triggered) so a caller can log/report what fired
        even when ``policy="redact"`` already fixed it up. When nothing is found, ``item_to_store is
        item`` (no copy is made) and ``scan.clean`` is ``True``.

    Raises:
        SecretPolicyError: ``policy="reject"`` and a secret was detected anywhere in the surface.
        ValueError: ``policy`` is not one of :data:`SECRET_POLICIES`.
    """
    if policy not in SECRET_POLICIES:
        raise ValueError(f"unknown secret_policy {policy!r}; expected one of {SECRET_POLICIES}")
    scan = detect_secrets(item_surface(item))
    if scan.clean:
        return item, scan
    if policy == "reject":
        raise SecretPolicyError(scan, item_id=getattr(item, "id", None))
    sanitized = replace(
        item,
        text=safe_text(getattr(item, "text", "") or ""),
        payload=redact_value(getattr(item, "payload", None) or {}),
        tags=[safe_text(str(t)) for t in (getattr(item, "tags", None) or [])],
    )
    # Verify the sanitization on the same surface the detection used, before anything is stored
    # (MXR-080-1882). Redaction previously covered strictly less of the surface than the scan did, so
    # an item could be detected dirty, "redacted", and stored with the secret intact -- and nothing
    # in the write path ever looked again. This closes that as a class rather than as three cases: if
    # any future payload shape is reachable by item_surface but not by redact_value, the write fails
    # here instead of leaking.
    residual = detect_secrets(item_surface(sanitized))
    if not residual.clean:
        raise SecretPolicyError(residual, item_id=getattr(item, "id", None))
    return sanitized, scan
