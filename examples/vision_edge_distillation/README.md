# Vision edge distillation: GPU training with CPU verification

This example explores whether a compact student can learn an embedding produced by a larger
vision model and later run without that model. Results depend on the artifact, dataset snapshot,
hardware, and dependency versions. This tracked page intentionally publishes no 0.8.0 performance
number; a run is evidence only when its artifact digests and environment are retained with its
receipt. The scripts pin both Hugging Face repositories to full immutable commit revisions and record
those revisions plus dataset fingerprints in ``metrics.json``.

The directory demonstrates a **laptop + pool** topology: a GPU training pass distils the teacher's
feature space into a compact student, and a separate CPU process verifies the resulting files.

## The technique — feature distillation

`distill_clip_features.py` (run on a CUDA GPU): a compact CNN learns to mimic CLIP's image embedding
on CIFAR-10 via a cosine loss, then classifies zero-shot
through CLIP's *frozen* text head. The student carries CLIP's geometry; it never sees CLIP at inference.

`verify_on_laptop.py` (run on a CPU): authenticates the retrieved files, restricts deserialization to
tensor weights, and reports the measurements from that run. Its output is not a release claim unless
the release evidence bundle records the exact inputs and environment.

## Reproduce

```bash
# on a CUDA GPU (e.g. a cheap vast.ai instance):
pip install torch transformers datasets pillow
python distill_clip_features.py           # writes student.pt, student_head.pt, metrics.json

# copy those three files next to verify_on_laptop.py. Obtain their SHA-256 digests
# from the artifact producer over the same trusted channel as the release evidence,
# then verify on the CPU:
python verify_on_laptop.py \
  --student-sha256 <student.pt-sha256> \
  --head-sha256 <student_head.pt-sha256> \
  --metrics-sha256 <metrics.json-sha256>
```

## What this establishes

The example establishes a reproducible *procedure* and trust boundary, not a fixed performance
conclusion. Use the generated measurements to decide whether a particular student satisfies a
particular deployment target.
