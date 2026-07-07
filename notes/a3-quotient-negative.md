# CARD A3-a: translation-quotient leaf research spike -- negative result

## What was built

`mixle/models/quotient.py`:

- `TranslationQuotientLeaf`: conv stack (two 3x3 same-padding conv layers, 3->16->32 channels) -> global
  average pool -> linear -> softmax, wrapped as a `NeuralCategorical` leaf. Declares its symmetry group as
  `leaf.group == "translation"` (also `leaf.declared_group()`).
- `UnpooledConvLeaf`: the same-capacity baseline -- identical conv stack, but flatten + dense instead of
  global pooling, no declared group.
- `shift_image_batch`: zero-padded integer pixel shift, used both to test the invariance property and to
  build the "corrupted" (shifted) test set for the robustness comparison.

Parameter counts on the 4-class CIFAR-10 setup below: quotient leaf 5,220 params, unpooled baseline
136,164 params (the baseline's dense layer dominates: 32*32*32 -> 4). This is the expected shape for a
"quotient" leaf -- pooling trades capacity for invariance -- but it also means the comparison is NOT
matched on parameter count, only on conv depth/width, which is what the card asked for.

## Dataset

CIFAR-10, loaded via `datasets.load_dataset("cifar10")` from the local HuggingFace cache
(`~/.cache/huggingface/datasets/cifar10`), fully offline, no network fetch needed. A 4-class subset
(airplane, automobile, bird, cat) was used: 300 images/class train (1200 total), 80 images/class test
(320 total), pixels scaled to [0, 1], each image `(3, 32, 32)`.

## Measured results (from an actual training run, `torch` 2.12.1 CPU)

```
train (1200, 3, 32, 32) test (320, 3, 32, 32)
param counts: quotient 5220 baseline 136164
max abs logp diff under shift(2,3) [quotient, full test set]: 1.3409238 mean: 0.20709479
max abs logp diff under shift(2,3) [baseline, full test set]: 25.870012 mean: 5.9330955
1/4-data test accuracy: quotient=0.4250 baseline=0.5188
full-data test accuracy: quotient=0.5219 baseline=0.6438
shifted-test accuracy (models trained on full data): quotient=0.4719 baseline=0.4437
clean-test accuracy (models trained on full data):   quotient=0.5219 baseline=0.6438
```

### (a) Invariance property -- holds, as expected

The quotient leaf's log-density is near-invariant under a real `(dy=2, dx=3)` pixel shift: mean absolute
log-density difference 0.207 nats (max 1.34 nats) across the full test set, versus 5.93 nats mean (max
25.87) for the unpooled baseline -- roughly 28x smaller. This confirms the architectural claim: global
average pooling after same-padding convs makes `log_density(x) ~= log_density(shift(x))`, and the
unpooled baseline has no such property. This part of the spike is a clean **win** and is pinned down by
`mixle/tests/quotient_leaf_test.py::test_quotient_leaf_log_density_is_shift_invariant_on_real_inputs` and
`::test_baseline_leaf_lacks_shift_invariance`.

### (b) Sample efficiency (accuracy at 1/4 training data) -- LOSS

Quotient: 0.4250, baseline: 0.5188. The unpooled baseline is ~9 points more accurate at 1/4 data. The
quotient leaf does not win here.

### (c) Robustness (accuracy under a small-translation corruption of the test set) -- narrow, low-value win

Quotient: 0.4719 vs baseline: 0.4437 under the shift corruption (both models trained on full data) -- the
quotient leaf is ~3 points more robust in this one narrow sense. But this has to be read against the
absolute accuracy levels: the baseline's *clean* accuracy (0.6438) is so much higher than either model's
shifted accuracy that the baseline is still better in absolute terms even after the corruption partially
erodes its edge (0.6438 -> 0.4437, a 20-point drop, versus the quotient leaf's much smaller drop from
0.5219 -> 0.4719). So "robustness" as a relative-degradation story is real, but "robustness" as an absolute
accuracy story under corruption is a near-tie that does not clearly favor the quotient leaf either.

## Kill-criterion verdict: LOSS -- documented negative result, per the card's stop rule

The card's kill criterion requires the quotient leaf to beat the baseline on sample efficiency **or**
robustness. Sample efficiency is a clear loss (0.425 vs 0.519). Robustness is, at best, a narrow and
qualified win in one framing (delta under corruption) that does not survive when judged on absolute
post-corruption accuracy. Given the ambiguity on (c) and the clear loss on (b), this spike does not meet
the bar for a "win" the card asks for, and per the card's explicit instruction this is treated as the
valid, expected negative outcome: the note is written, the implementation and tests are kept (the
invariance property itself is real and tested), and no further tuning was done to try to force a win.

## Why this is plausible, not just a bug

The global-average-pool leaf has ~26x fewer parameters than the unpooled baseline (5,220 vs 136,164) for
the same conv stack -- most of the baseline's capacity lives in its enormous flatten+dense layer
(32*32*32 = 32,768 inputs to a 4-way linear head). On a small 4-class subset with only 1,200 training
images, that extra capacity appears to help the baseline more than translation invariance helps the
quotient leaf: CIFAR object classes are not purely translation-nuisance tasks (objects vary in scale, pose,
and background as much as position), and centered, uncropped CIFAR images likely already have limited
position variance for the baseline to overfit to. A convolutional feature extractor already gives
"quasi-invariant" partial credit through weight sharing before pooling; the marginal value of forcing full
invariance via global pooling, at the cost of throwing away all spatial/positional information the linear
head could have used, is real but here it is outweighed by the capacity loss on this dataset/task.

## Scope note

Per the card, no further tuning was attempted after this result (no capacity ladder sweep, no alternate
pooling strategy, no different dataset) -- this is the one specified real-dataset comparison, and the
result is reported as measured.
