"""The fuzzy nerve of the learned cover, and map-fidelity receipts (design review R3/R2, partial).

The fitted mixture is a parametrized cover of the data manifold: components are cover elements and
the posterior is a partition of unity subordinate to it. By the nerve theorem, the topology of the
data (for a good cover) is the homotopy type of the cover's NERVE -- the simplicial complex whose
k-simplices are (k+1)-wise overlapping components. :func:`fuzzy_nerve` computes its weighted 1- and
2-skeleton directly from the posteriors (this is soft Mapper with an EM-learned cover), and
:func:`nerve_report` turns it into receipts a 2-D map cannot give on its own:

* **holes** -- an unfilled cycle of strongly-overlapping components (e.g. a ring-shaped manifold)
  is a loop in the data's topology. A plane embedding may render it faithfully or may distort it;
  either way the user should KNOW the loop exists rather than infer it from blob positions.
* **disconnection** -- a cover that splits into multiple connected pieces.

Cycle classification is deliberately conservative: a 3-cycle is "filled" exactly when its triple
overlap is strong (the 2-simplex is present); longer cycles are reported as candidate holes without
attempting a full triangulation check (exact filling is a combinatorial problem; the honest output
is "here is a cycle no strong triple fills directly").

:func:`embedding_health` is the RENDERING-fidelity receipt: trustworthiness/continuity of the map's
neighborhoods against the model affinity that produced it. Note precisely what this does and does
not audit -- it catches a layout that misrepresents the model (rendering failure), NOT a model that
misrepresents the data (model misfit, which remains the design review's open R2 item).
"""

from __future__ import annotations

import numpy as np

__all__ = ["component_tree", "embedding_health", "fuzzy_nerve", "model_fit_health", "nerve_report"]


def fuzzy_nerve(z: np.ndarray, *, edge_threshold: float = 0.02, triangle_threshold: float = 0.02) -> dict:
    """Weighted 1- and 2-skeleton of the cover's nerve, from the posteriors alone.

    Edge weight ``w(k,l) = sum_i z_ik z_il / min(mass_k, mass_l)`` -- the co-claimed fraction of the
    smaller component's mass (1 when one component's points are entirely co-claimed by the other,
    0 when they never co-claim). Triangle weight is the same with a triple product. Simplices at or
    above their threshold are "strong" and drive :func:`nerve_report`; all nonzero weights are
    returned so thresholds are inspectable choices, not hidden ones.
    """
    z = np.asarray(z, dtype=np.float64)
    k_count = z.shape[1]
    masses = z.sum(axis=0)
    safe = np.maximum(masses, 1.0e-12)

    co = z.T @ z
    edges: dict[tuple[int, int], float] = {}
    for a in range(k_count):
        for b in range(a + 1, k_count):
            w = float(co[a, b] / min(safe[a], safe[b]))
            if w > 0.0:
                edges[(a, b)] = w

    triangles: dict[tuple[int, int, int], float] = {}
    strong_pairs = {e for e, w in edges.items() if w >= edge_threshold}
    for a in range(k_count):
        for b in range(a + 1, k_count):
            if (a, b) not in strong_pairs:
                continue  # a strong triangle needs strong faces; skip the O(K^3) full sweep
            for c in range(b + 1, k_count):
                if (a, c) not in strong_pairs or (b, c) not in strong_pairs:
                    continue
                w = float(np.einsum("i,i,i->", z[:, a], z[:, b], z[:, c]) / min(safe[a], safe[b], safe[c]))
                if w > 0.0:
                    triangles[(a, b, c)] = w

    return {
        "masses": masses,
        "edges": edges,
        "triangles": triangles,
        "edge_threshold": float(edge_threshold),
        "triangle_threshold": float(triangle_threshold),
    }


def _independent_cycles(nodes: list[int], edges: list[tuple[int, int]]) -> tuple[int, list[list[int]]]:
    """Connected-component count and one representative cycle per independent loop (spanning-forest
    construction: every non-tree edge closes exactly one cycle, recovered through tree parents)."""
    adj: dict[int, list[int]] = {v: [] for v in nodes}
    tree_edges = set()
    parent: dict[int, int | None] = {}
    n_components = 0
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    for root in nodes:
        if root in parent:
            continue
        n_components += 1
        parent[root] = None
        stack = [root]
        while stack:
            v = stack.pop()
            for u in adj[v]:
                if u not in parent:
                    parent[u] = v
                    tree_edges.add((min(v, u), max(v, u)))
                    stack.append(u)

    def path_to_root(v: int) -> list[int]:
        path = [v]
        while parent[path[-1]] is not None:
            path.append(parent[path[-1]])
        return path

    cycles = []
    for a, b in edges:
        if (min(a, b), max(a, b)) in tree_edges:
            continue
        pa, pb = path_to_root(a), path_to_root(b)
        common = set(pa) & set(pb)
        cut_a = next(i for i, v in enumerate(pa) if v in common)
        anchor = pa[cut_a]
        cut_b = pb.index(anchor)
        cycles.append(pa[: cut_a + 1] + pb[:cut_b][::-1])
    return n_components, cycles


def nerve_report(nerve: dict) -> dict:
    """Topology receipts from a :func:`fuzzy_nerve`: connected pieces, cycles, and candidate holes.

    A hole is an independent cycle of strong edges not directly filled by a strong 2-simplex
    (3-cycles are checked exactly; longer cycles are conservatively reported as candidates). The
    ``diagnosis`` strings are the user-facing half: a loop in the cover is real data topology that
    a 2-D layout may distort silently.
    """
    strong_edges = [e for e, w in nerve["edges"].items() if w >= nerve["edge_threshold"]]
    strong_triangles = {t for t, w in nerve["triangles"].items() if w >= nerve["triangle_threshold"]}
    k_count = len(nerve["masses"])
    n_components, cycles = _independent_cycles(list(range(k_count)), strong_edges)

    holes = []
    for cycle in cycles:
        filled = len(cycle) == 3 and tuple(sorted(cycle)) in strong_triangles
        if not filled:
            holes.append(sorted(cycle))

    diagnosis = []
    if n_components > 1:
        diagnosis.append(
            f"the component cover is disconnected into {n_components} pieces: inter-piece distances in the "
            "map reflect only the layout, not measured overlap."
        )
    for hole in holes:
        diagnosis.append(
            f"the cover contains an unfilled cycle through components {hole}: the data topology has a loop "
            "(ring/period-like structure) that a 2-D layout may distort -- verify it deliberately rather "
            "than reading it off blob positions."
        )

    return {
        "n_components": n_components,
        "n_strong_edges": len(strong_edges),
        "cycles": [sorted(c) for c in cycles],
        "holes": holes,
        "diagnosis": diagnosis,
    }


def component_tree(nerve: dict) -> list[dict]:
    """Single-linkage merge tree over components by nerve edge weight -- the hierarchy skeleton.

    Merges are emitted strongest-overlap-first: ``[{"a": frozenset, "b": frozenset, "weight": w,
    "merged": frozenset}, ...]``. Cutting the tree at any weight gives coarse super-components
    (:class:`mixle.utils.hvis.front.Map` uses it for zoom groups); the merge order is itself a
    receipt -- which regimes are almost one regime.
    """
    k_count = len(nerve["masses"])
    parent = list(range(k_count))

    def find(v: int) -> int:
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    merges = []
    for (a, b), w in sorted(nerve["edges"].items(), key=lambda item: (-item[1], item[0])):
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        group_a = frozenset(v for v in range(k_count) if find(v) == ra)
        group_b = frozenset(v for v in range(k_count) if find(v) == rb)
        parent[ra] = rb
        merges.append({"a": group_a, "b": group_b, "weight": float(w), "merged": group_a | group_b})
    return merges


def model_fit_health(
    mix_model,
    data,
    *,
    holdout=None,
    field_weights=None,
    coverage_q: float = 0.9,
    merged_sep_threshold: float | None = None,
    shattered_weight: float = 0.5,
    min_component_points: int = 20,
) -> dict:
    """The model<->data receipt (design review R2, second half): does the FITTED MODEL describe the
    data it is about to be a map of? Measured from the model's own residual structure -- no raw
    feature space is assumed, which is the whole point of HViS.

    * **fiber calibration** -- per component, the squared Mahalanobis of its dominant points'
      whitened fiber coordinates should look chi-squared: the fraction inside the ``coverage_q``
      ball is compared against ``coverage_q``. A large gap means the component's shape claim is
      wrong (too wide, too narrow, or mis-shaped).
    * **merged-regime detector** -- a deterministic 2-means split (top-PC sign init) of each
      component's dominant fiber coordinates; a separation ratio above the threshold with a
      non-trivial minority says one component is covering what the data treats as two regimes
      (K too small). The threshold has a derivation plus a measured finite-sample correction: for
      a UNIMODAL normal the population statistic is ``2 E|x| / sqrt(1 - 2/pi) ~ 2.65`` regardless
      of scale, but at n=40 sample noise inflates it to ~3.4 (observed), so the default threshold
      is ``2.65 + 6/sqrt(n)`` -- ~3.6 at n=40, tightening toward the population value as n grows.
      Two unit-variance regimes 4 sigma apart score ~4.0 either way. Pass an explicit
      ``merged_sep_threshold`` to pin it.
    * **shattered detector** -- nerve edges with weight >= ``shattered_weight`` are near-duplicate
      components claiming largely the same points (K too large).
    * **held-out check** -- with ``holdout`` data, a mean log-density drop > 1 nat vs training is
      flagged (memorization / drift).
    """
    from scipy.stats import chi2

    from mixle.utils.hvis.direct import component_fiber_coords

    data = list(data)
    z, us, _labels, _t = component_fiber_coords(mix_model, data, field_weights=field_weights)
    k_count = z.shape[1]
    dominant = z.argmax(axis=1)

    components = []
    diagnosis = []
    for k in range(k_count):
        mine = us[k][dominant == k]
        entry: dict = {"n_points": int(len(mine)), "coverage": None, "merged_separation": None}
        if len(mine) >= min_component_points:
            mu = mine.mean(axis=0)
            centered = mine - mu
            dim = mine.shape[1]
            cov = centered.T @ centered / max(len(mine) - 1, 1) + 1.0e-9 * np.eye(dim)
            m2 = np.einsum("ij,jk,ik->i", centered, np.linalg.pinv(cov), centered)
            coverage = float(np.mean(m2 <= chi2.ppf(coverage_q, df=dim)))
            entry["coverage"] = coverage
            if abs(coverage - coverage_q) > 0.15:
                diagnosis.append(
                    f"component {k}: fiber calibration off (coverage {coverage:.2f} vs nominal {coverage_q}) -- "
                    "its shape claim does not match its points."
                )

            # deterministic 2-means on the top principal axis: split by sign, then Lloyd iterations
            vals, vecs = np.linalg.eigh(cov)
            axis = vecs[:, np.argmax(vals)]
            proj = centered @ axis
            assign = proj > 0.0
            for _ in range(15):
                if assign.all() or (~assign).all():
                    break
                c1, c0 = float(proj[assign].mean()), float(proj[~assign].mean())
                new_assign = np.abs(proj - c1) < np.abs(proj - c0)
                if bool(np.all(new_assign == assign)):
                    break
                assign = new_assign
            if 0 < int(assign.sum()) < len(proj):
                minority = min(float(assign.mean()), float(1.0 - assign.mean()))
                within_var = float(
                    np.average([proj[assign].var(), proj[~assign].var()], weights=[assign.mean(), 1 - assign.mean()])
                )
                sep = abs(float(proj[assign].mean() - proj[~assign].mean())) / max(np.sqrt(within_var), 1.0e-12)
                entry["merged_separation"] = sep
                threshold = (
                    merged_sep_threshold if merged_sep_threshold is not None else 2.65 + 6.0 / np.sqrt(float(len(mine)))
                )
                if sep > threshold and minority >= 0.2:
                    diagnosis.append(
                        f"component {k}: looks like TWO merged regimes (2-means separation {sep:.1f}, minority "
                        f"{minority:.0%}) -- the mixture may need more components here."
                    )
        components.append(entry)

    nerve = fuzzy_nerve(z)
    shattered = [(list(e), float(w)) for e, w in nerve["edges"].items() if w >= shattered_weight]
    for pair, w in shattered:
        diagnosis.append(
            f"components {pair}: near-duplicates (overlap weight {w:.2f}) -- they claim largely the same "
            "points; the mixture may have too many components here."
        )

    holdout_drop = None
    if holdout is not None:
        from mixle.utils.hvis.stream import _mean_log_density

        train_ll = _mean_log_density(mix_model, data)
        hold_ll = _mean_log_density(mix_model, list(holdout))
        holdout_drop = float(train_ll - hold_ll)
        if holdout_drop > 1.0:
            diagnosis.append(
                f"held-out mean log-density is {holdout_drop:.2f} nats below training -- the fit does not "
                "generalize to the data the map will be read against."
            )

    return {
        "components": components,
        "shattered_pairs": shattered,
        "holdout_drop_nats": holdout_drop,
        "diagnosis": diagnosis,
    }


def embedding_health(
    coords: np.ndarray,
    mix_model,
    data,
    *,
    affinity="auto",
    k: int = 10,
    field_weights=None,
    evidence_cap: float | None = 1.0,
    max_rows: int = 400,
    seed: int = 0,
) -> dict:
    """Rendering-fidelity receipt: do the map's neighborhoods agree with the model affinity?

    Standard trustworthiness (are map-neighbors genuinely close under the model?) and continuity
    (are model-neighbors kept close in the map?), computed on a row subsample. This audits the
    LAYOUT against the MODEL -- a low score means the picture misrepresents the affinities that
    produced it (bad init, unconverged optimizer, non-embeddable topology). It does NOT audit the
    model against the raw data; that receipt is still open (design review R2).
    """
    from mixle.utils.hvis.affinity import (
        _affinity_factors,
        _posteriors_and_loglikes,
        _resolve_affinity,
        log_affinity_block,
    )

    coords = np.asarray(coords, dtype=np.float64)
    data = list(data)
    resolved = _resolve_affinity(affinity, mix_model, data, field_weights)
    if isinstance(resolved, str):
        z, ll = _posteriors_and_loglikes(mix_model, data=data)
        factors = _affinity_factors(z, ll, resolved)
    else:
        factors = _affinity_factors(None, None, resolved)

    n = coords.shape[0]
    rng = np.random.RandomState(seed)
    idx = np.arange(n) if n <= max_rows else np.sort(rng.choice(n, size=max_rows, replace=False))
    m = len(idx)
    k = min(int(k), m - 2)

    log_s = log_affinity_block(factors, idx, idx, evidence_cap)
    np.fill_diagonal(log_s, -np.inf)
    d2 = np.square(coords[idx][:, None, :] - coords[idx][None, :, :]).sum(axis=2)
    np.fill_diagonal(d2, np.inf)

    # rank matrices: rank_data[i, j] = position of j in i's model-affinity ordering (1 = nearest)
    order_data = np.argsort(-log_s, axis=1, kind="stable")
    order_map = np.argsort(d2, axis=1, kind="stable")
    rank_data = np.empty((m, m), dtype=np.int64)
    rank_map = np.empty((m, m), dtype=np.int64)
    rows = np.arange(m)[:, None]
    rank_data[rows, order_data] = np.arange(m)[None, :] + 1
    rank_map[rows, order_map] = np.arange(m)[None, :] + 1

    knn_data = rank_data <= k
    knn_map = rank_map <= k
    norm = 2.0 / (m * k * (2.0 * m - 3.0 * k - 1.0))

    trust_pen = np.where(knn_map & ~knn_data, rank_data - k, 0).sum(axis=1).astype(np.float64)
    cont_pen = np.where(knn_data & ~knn_map, rank_map - k, 0).sum(axis=1).astype(np.float64)
    trustworthiness = float(1.0 - norm * trust_pen.sum())
    continuity = float(1.0 - norm * cont_pen.sum())

    diagnosis = []
    if trustworthiness < 0.85:
        diagnosis.append(
            f"trustworthiness {trustworthiness:.2f}: the map shows neighbor pairs the model does not "
            "support -- treat visual proximity with suspicion (bad init, unconverged refine, or a "
            "topology 2-D cannot hold)."
        )
    if continuity < 0.85:
        diagnosis.append(
            f"continuity {continuity:.2f}: genuinely-close pairs (under the model) were torn apart in the "
            "map -- clusters or fibers may be split visually that the model considers one region."
        )

    return {
        "trustworthiness": trustworthiness,
        "continuity": continuity,
        "per_point_trust_penalty": trust_pen,
        "per_point_continuity_penalty": cont_pen,
        "k": k,
        "n_sampled": m,
        "diagnosis": diagnosis,
    }
