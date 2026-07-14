"""E10 acceptance: the occupancy receipt on NATURAL data, not random-token streams.

K3's headline (49 occupied cells, flat from 4k to 64k context) was measured on random-token
streams through a model trained on a 65-symbol synthetic task -- the falsification's own caveat
says richer data will occupy more cells and the mechanism only stays honest if occupancy is
MEASURED, not assumed. This streams real English prose (the repository's own README + docs,
byte-level, vocab 256) through a briefly-TBPTT-trained spine and records the receipt at growing
context lengths.

Acceptance (pre-stated): occupancy at 64k context stays BOUNDED AWAY from capacity (< 80% of
max_cells with zero dropped tokens) and grows sublinearly with context (occupied(64k) <
2 x occupied(4k)); otherwise the O(occupied cells) claim does not transfer to natural data at
this scale -- record the numbers honestly either way.
"""

import pathlib
import time

import numpy as np
import torch

from mixle.experimental.quantized_key_attention import QuantizedKeyAttentionSpine

torch.manual_seed(0)
ROOT = pathlib.Path(__file__).resolve().parents[2]
CHUNK, WINDOW, MAX_CELLS = 32, 16, 256
TRAIN_STEPS, BATCH = 300, 4


def corpus_bytes() -> np.ndarray:
    parts = [ROOT / "README.md"]
    parts += sorted((ROOT / "docs").rglob("*.rst"))
    blob = b"\n\n".join(p.read_bytes() for p in parts if p.is_file())
    return np.frombuffer(blob, dtype=np.uint8).astype(np.int64)


DATA = corpus_bytes()
print(f"corpus: {len(DATA):,} bytes from README + docs/*.rst")
HELD_OUT_START = len(DATA) // 2  # train on the first half, stream occupancy on the second

spine = QuantizedKeyAttentionSpine(
    256, d_model=32, n_layer=2, n_head=2, window=WINDOW, n_blocks=4, codes_per_block=8, max_cells=MAX_CELLS
)
opt = torch.optim.Adam(spine.parameters(), lr=3e-4)
rng = np.random.RandomState(0)

print(f"TBPTT training: {TRAIN_STEPS} steps x batch {BATCH} x chunk {CHUNK} on the first half")
spine.train()
for step in range(1, TRAIN_STEPS + 1):
    starts = rng.randint(0, HELD_OUT_START - 4 * CHUNK - 1, BATCH)
    state = spine.init_state(BATCH)
    total = None
    for c in range(4):  # 4-chunk TBPTT segments
        x = torch.as_tensor(np.stack([DATA[s + c * CHUNK : s + (c + 1) * CHUNK] for s in starts]))
        y = torch.as_tensor(np.stack([DATA[s + c * CHUNK + 1 : s + (c + 1) * CHUNK + 1] for s in starts]))
        state, loss = spine.step(state, (x, y))
        total = loss if total is None else total + loss
        state = spine.detach(state)
    opt.zero_grad()
    total.backward()
    opt.step()
    if step % 100 == 0:
        print(f"  step {step:4d}  segment loss {float(total) / 4:.3f}")

print("\noccupancy receipts on held-out natural text (single stream, no grad):")
spine.eval()
receipts = {}
for n_ctx in (4096, 16384, 65536):
    stream = DATA[HELD_OUT_START : HELD_OUT_START + n_ctx + 1]
    state = spine.init_state(1)
    t0 = time.perf_counter()
    with torch.no_grad():
        for start in range(0, n_ctx, CHUNK):
            x = torch.as_tensor(stream[start : start + CHUNK])[None]
            y = torch.as_tensor(stream[start + 1 : start + CHUNK + 1])[None]
            state, _ = spine.step(state, (x, y))
    receipt = spine.occupancy_receipt(state)
    receipts[n_ctx] = receipt
    wall = time.perf_counter() - t0
    print(
        f"  n={n_ctx:6d}  occupied/layer={receipt['occupied_cells_per_layer']}  "
        f"capacity={receipt['capacity']}  possible={receipt['possible_cells']}  "
        f"drops={receipt['dropped_tokens']}  ({wall:.1f}s)"
    )

occ4 = max(receipts[4096]["occupied_cells_per_layer"])
occ64 = max(receipts[65536]["occupied_cells_per_layer"])
drops = receipts[65536]["dropped_tokens"]
bounded = occ64 < 0.8 * MAX_CELLS and drops == 0
sublinear = occ64 < 2 * occ4
print(
    f"\nacceptance: occupied(64k)={occ64} (< {0.8 * MAX_CELLS:.0f} and 0 drops: {bounded}); "
    f"occupied(4k)={occ4}, sublinear growth (< 2x): {sublinear} -> "
    f"{'PASS' if bounded and sublinear else 'FAIL'}"
)
