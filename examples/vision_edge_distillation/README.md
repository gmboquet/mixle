# Vision edge distillation: closing the gap with a rented GPU, verified on a laptop

`foundation_to_edge.py` reports an honest negative: a **vision** foundation capability (CLIP) does **not**
distil onto raw pooled pixels — a kilobyte pixel-student recovers ~14%, and even a from-scratch CNN
trained on a laptop (3k images, MPS) reaches only ~40%. The capability lives in CLIP's web-scale
pretrained features, which a small student on little data cannot reconstruct.

This directory closes that gap the honest way — the **laptop + pool** topology the workplan describes: a
one-time GPU training pass distils CLIP's *feature space* (not just its labels) into a compact student
that then runs on the laptop with no CLIP and no GPU.

## The technique — feature distillation

`distill_clip_features.py` (run on a CUDA GPU): a compact CNN (1.28M params) learns to **mimic CLIP's
512-d image embedding** on all 50k CIFAR-10 images via a cosine loss, then classifies **zero-shot**
through CLIP's *frozen* text head. The student carries CLIP's geometry; it never sees CLIP at inference.

`verify_on_laptop.py` (run on the laptop, CPU): loads the retrieved 5 MB student and reproduces its
accuracy locally — the round-trip receipt.

## The receipt (measured, not asserted)

Distilled on a rented RTX 3060 (vast.ai), then verified on a MacBook CPU:

| student | accuracy | notes |
|---|---|---|
| pooled-pixel (KB, torch-free) | 0.14 | the honest negative from `foundation_to_edge.py` |
| from-scratch CNN, 3k imgs, laptop MPS | 0.40 | label distillation, too little data |
| **feature-distilled, 50k imgs, GPU** | **0.82** | 1.28M params, **5.1 MB**, 93% of CLIP's zero-shot |

- CLIP zero-shot teacher: **0.888** on full CIFAR-10 test.
- Student: **0.8165** (GPU) / **0.822** (laptop CPU re-verify) — **93% retention**, **118× smaller** than
  CLIP (5 MB vs 605 MB), classifies 2000 images in 1.8 s on CPU.
- Training: 40 epochs, ~5 min on the RTX 3060. **Total rented-compute cost: $0.04.**

## Reproduce

```bash
# on a CUDA GPU (e.g. a cheap vast.ai instance):
pip install torch transformers datasets pillow
python distill_clip_features.py           # writes student.pt, student_head.pt, metrics.json

# copy those three files next to verify_on_laptop.py, then on the laptop (CPU):
python verify_on_laptop.py
```

## What this establishes

Foundation-capability-on-laptop and edge distillation are **here**, with the boundary mapped:

- **Text**: distils to a torch-free KB student directly on the laptop (see `foundation_to_edge.py`).
- **Vision**: needs one GPU pass to feature-distil (pennies on rented compute), then runs on the
  laptop/edge in a few MB at 93% retention. The "keeps the encoder" fallback is no longer the only
  option — a distilled student is.
