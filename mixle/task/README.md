# `mixle.task` — compact local models for task replacement and routing

Turn an expensive model call, slow rule, or human-labeled workflow into a compact local model for one
task. The local model answers when it is calibrated to do so, escalates uncertain inputs to the teacher,
and harvests those escalations as targeted labels for later improvement. The pieces are independent, but
compose into one measurable distill-calibrate-serve loop.

## The loop

```
active labeling -> teacher -> distill -> tune -> calibrate + density gate -> cascade -> harvest

The loop uses DOE to reduce labeling, student training to replace repeated teacher calls, conformal
calibration to decide between local answers and escalation, and harvested traffic to improve the next
student.
```

## What each piece does

| Module | Purpose |
| --- | --- |
| `distill` / `distill_records` | A teacher labels data; a compact student (text **or** structured record) learns to match and returns a callable `TaskModel`. |
| `distill_structured` | Distill a teacher into a **structured probabilistic** student — a learned dependency network (`learn_structure`), not an MLP. Classifies generatively (`argmax P(fields, label)`), exposes posterior confidence for calibration/cascade workflows, stores interpretable `meta["edges"]`, and runs without torch. `n_components>1` gives a latent-regime mixture-of-trees. |
| `active_distill` | **DoE for the labeling decision** — query the teacher only for the most informative examples. Same quality, far fewer paid labels. |
| `tune_recipe` | **DoE for training cost** — Bayesian-optimize the student recipe; optional compute penalty favors the lowest-cost recipe that still matches. |
| `llm_labeler` + `OpenAICompatLLM` | Make an LLM the teacher (Ollama/vLLM/TGI/hosted, stdlib-only client). |
| `CalibratedTaskModel` | Conformal prediction sets produce a `decide()` method that returns a label only when calibrated to do so, otherwise `ESCALATE`. |
| `DensityGate` | A generative `p(x)` over inputs — escalate inputs outside the training distribution. |
| `Cascade` | Serve locally when confident, escalate otherwise; report realized savings; harvest escalations as targeted labels for re-distillation. |
| `CostModel` / `break_even_volume` / `recommend_route` | Cost arithmetic for deciding whether GPU, label, and training spend pays back at the expected request volume. |
| `design_model` | An LLM proposes a mixle model structure from data; mixle **validates by fitting** it, falling back to the heuristic `recommend_model`. |
| `TaskModel.save` / `load` | Durable artifact (manifest + safetensors): load in any process and call. |

## Quickstart

```python
from mixle.task import distill, CalibratedTaskModel, Cascade, CostModel

teacher = my_llm_or_rule                      # any callable: texts -> labels
student = distill(teacher, unlabeled_texts)   # compact local model
gate    = CalibratedTaskModel(student, alpha=0.1).calibrate(cal_texts, teacher(cal_texts))

casc = Cascade(gate, teacher, cost=CostModel(c_frontier=0.01, c_local=1e-5))
casc.serve(traffic)                           # local when confident, teacher otherwise
print(casc.report()["savings_vs_frontier"])   # realized $ saved
```

See `examples/task_distill_example.py`, `examples/task_cascade_economics_example.py`, and
`examples/task_llm_active_example.py`.
