# `mixle.task` — small local models that replace hardcoded logic, trained cheaply

Turn an expensive model call (a frontier LLM, a slow rule, a human) into a tiny local model that does **one
task** fast and for ~free — and only pay the expensive model for the cases the small one can't handle. The
pieces are independent but compose into one cost-cutting loop.

## The loop

```
active labeling ──> LLM teacher ──> distill ──> tune ──> calibrate + density gate ──> Cascade ──> harvest ──┐
   (DoE: fewest      (label-          (tiny      (DoE:      (honest "answer vs          (serve:         (re-distill, ┘
    labels)           constrained      student)   cheapest   escalate" + real p(x))      local-when-     cheaper with use)
                      LLM)                         recipe)                                confident)
```

## What each piece does

| Module | Purpose |
| --- | --- |
| `distill` / `distill_records` | A teacher labels data; a tiny student (text **or** structured record) learns to match → a callable `TaskModel`. |
| `active_distill` | **DoE for the labeling decision** — query the teacher only for the most informative examples. Same quality, far fewer paid labels. |
| `tune_recipe` | **DoE for training cost** — Bayesian-optimize the student recipe; optional compute penalty finds the cheapest recipe that still matches. |
| `llm_labeler` + `OpenAICompatLLM` | Make an LLM the teacher (Ollama/vLLM/TGI/hosted, stdlib-only client). |
| `CalibratedTaskModel` | Conformal prediction sets → an honest `decide()` that returns a label only when confident, else `ESCALATE`. The softmax isn't a probability; conformal makes the decision *guaranteed*. |
| `DensityGate` | A real generative `p(x)` over inputs — escalate inputs the model has never seen (what a softmax structurally can't detect). |
| `Cascade` | Serve local-when-confident, escalate-else; report **realized dollars saved**; harvest escalations as free targeted labels to re-distill (cheaper with use). |
| `CostModel` / `break_even_volume` / `recommend_route` | The arithmetic: is the GPU/label spend worth it, and which route is cheapest at your volume? |
| `design_model` | An LLM proposes a mixle model structure from data; mixle **validates by fitting** it, falling back to the heuristic `recommend_model`. |
| `TaskModel.save` / `load` | Durable artifact (manifest + safetensors): load in any process and call. |

## Quickstart

```python
from mixle.task import distill, CalibratedTaskModel, Cascade, CostModel

teacher = my_llm_or_rule                      # any callable: texts -> labels
student = distill(teacher, unlabeled_texts)   # tiny local model
gate    = CalibratedTaskModel(student, alpha=0.1).calibrate(cal_texts, teacher(cal_texts))

casc = Cascade(gate, teacher, cost=CostModel(c_frontier=0.01, c_local=1e-5))
casc.serve(traffic)                           # local when confident, teacher otherwise
print(casc.report()["savings_vs_frontier"])   # realized $ saved
```

See `examples/task_distill_example.py`, `examples/task_cascade_economics_example.py`, and
`examples/task_llm_active_example.py`.
