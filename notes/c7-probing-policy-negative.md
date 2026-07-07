# CARD PROBE-a: learned non-myopic probing policy vs myopic EIG -- negative result

## What was built

`mixle/task/probe_policy.py`:

- `myopic_eig_policy`: one-step-lookahead policy. Maintains an explicit logistic belief
  `p_target(cell)` from that cell's current noisy prospectivity read (`mixle/task/explore_world.py`),
  centered on the world's own known decision boundary (target cells' geology reads systematically
  +2.0 higher). EXPLOITs (drills) the most confident current candidate once its belief clears a
  0.65 threshold; otherwise EXPLOREs (surveys) the single most uncertain cell -- the textbook
  maximum-entropy expected-information-gain target.
- `head_to_head_probe`: runs the non-myopic side (the outcome-trained decomposer's fitted plan
  model, `mixle/task/outcome_decomposer.py`, CARD C2-a -- trained via expert iteration against
  VERIFIER-GROUNDED reward, i.e. terminal world score, no learned reward model) and the myopic EIG
  policy on the SAME held-out world seeds at matched budget, and reports which wins.

An earlier version of the myopic policy ranked every action by raw `expected_information_gain / cost`
across the whole action menu. That degenerated: drilling costs 5x surveying, and the raw EIG estimate
for a drill was never large enough to clear that 5x cost penalty, so the policy essentially never
drilled at all and scored *worse than random* (a real bug, not part of the reported result below --
fixed before this comparison was run, by switching to an explicit exploit/explore threshold instead
of a single per-cost ranking).

## Measured result

20 cells, 3 targets, budget 30, 20 held-out seeds (disjoint from every training seed used to fit the
outcome-trained decomposer):

```
non-myopic (outcome-trained decomposer) mean score: 2.00
myopic EIG policy mean score:                       2.25
```

## Kill-criterion verdict: LOSS -- myopic EIG wins, per the card's stop rule

The card's kill criterion: if the learned/outcome-trained non-myopic policy does not beat myopic EIG
on solve-rate at matched budget (averaged over >= 20 held-out seeds), record the negative result and
KEEP myopic. It does not beat it here (2.00 vs 2.25) -- per the card's explicit instruction this is
the valid, expected outcome, not a bug to chase: "myopic is often near-optimal; this spike exists to
find the cases where delayed/combinatorial payoff makes it fail, not to add [a learned policy] for
its own sake." The implementation and both policies are kept and tested (`myopic_eig_policy` is now
the good baseline other future spikes in this world should compare against); no further tuning of the
outcome-trained side was done to try to force a win.

## Why this is plausible, not just an artifact

The exploration world's reward structure rewards CORRECTLY IDENTIFYING targets, which is exactly the
kind of task myopic EIG is suited for: the value of information about any one cell does not
meaningfully depend on the order other cells are resolved in (little combinatorial/delayed payoff --
no action opens up or forecloses a DIFFERENT cell's value), so a policy that always chases the
single-step-best move has little to lose relative to one that optimizes for a whole plan's terminal
score. The outcome-trained decomposer also only controls the ORDER/MIX of action TYPES (survey vs
drill), delegating WHICH cell to the same kind of per-cost heuristic myopic EIG makes directly and
per-step -- so its "non-myopic" advantage is thinner here than it would be in a world where probes
have real delayed/combinatorial structure (e.g. a probe that only pays off in combination with a
later, specific other probe). This matches the card's own framing of where a non-myopic policy should
be expected to win, and this benchmark does not have that structure.

## Scope note

Per the card, this is reported after the first head-to-head, win or lose -- no further tuning,
alternate reward shaping, or additional training rounds were attempted to try to reverse the result.
