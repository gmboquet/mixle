"""Does E10's far field learn needle retrieval AT ALL, or is 500 referee examples just too few?

The bake-off's kill signal (needle 0.000 at d=16 while E3a manages 0.312) is ambiguous between "the
cell store structurally cannot support retrieval learning" and "VQ friction needs a bigger budget than
the referee grants" -- the original falsification needed ~160k examples (2500 steps x batch 64) for the
quantized arm, 2x the dense arm's, and the E7 referee provides 500 (batch 1). This probe trains ONE arm
(E10, needle suite, d=16, window=8: retrieval must cross the far field) with a falsification-scale
budget and prints the accuracy trajectory. Either outcome sharpens RESULTS.md: a rising curve means
"budget", a flat one means "structure".
"""

import numpy as np
import torch

from mixle.experimental.long_context_eval import needle_suite
from mixle.experimental.quantized_key_attention import QuantizedKeyAttentionSpine

VOCAB = 16
DISTANCE = 16
WINDOW = 8
CHUNK = 8
STEPS = 3000
BATCH = 8  # the referee streams batch 1; batching here is purely budget, the mechanism is unchanged


def make_batch(rng: np.random.RandomState) -> tuple[torch.Tensor, torch.Tensor]:
    xs, ys = zip(*[needle_suite(rng, distance=DISTANCE, vocab=VOCAB) for _ in range(BATCH)])
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def main() -> None:
    torch.manual_seed(0)
    spine = QuantizedKeyAttentionSpine(
        VOCAB, d_model=16, n_layer=2, n_head=2, window=WINDOW, n_blocks=2, codes_per_block=8, max_cells=32
    )
    opt = torch.optim.Adam(spine.parameters(), lr=1e-2)
    rng = np.random.RandomState(0)
    chance = float(np.log(VOCAB))

    for step in range(1, STEPS + 1):
        x, y = make_batch(rng)
        state = spine.init_state(BATCH)
        loss_total = x.new_zeros((), dtype=torch.float32)
        for start in range(0, x.shape[1], CHUNK):
            state, loss = spine.step(state, (x[:, start : start + CHUNK], y[:, start : start + CHUNK]))
            loss_total = loss_total + loss
        opt.zero_grad()
        loss_total.backward()
        opt.step()

        if step % 300 == 0 or step == 1:
            probe_losses, solved = [], []
            eval_rng = np.random.RandomState(99)
            with torch.no_grad():
                for _ in range(32):
                    xe, ye = needle_suite(eval_rng, distance=DISTANCE, vocab=VOCAB)
                    st = spine.init_state(1)
                    for start in range(0, xe.shape[1] - 1, CHUNK):
                        st, _ = spine.step(st, (xe[:, start : start + CHUNK], ye[:, start : start + CHUNK]))
                    _, pl = spine.step(st, (xe[:, -1:], ye[:, -1:]))
                    probe_losses.append(float(pl))
                    solved.append(float(pl) < 0.5 * chance)
            receipt = spine.occupancy_receipt(st)
            print(
                f"step {step:5d}  needle_acc={np.mean(solved):.3f}  probe_loss={np.mean(probe_losses):.2f}"
                f"/{chance:.2f}  occupied={receipt['occupied_cells_per_layer']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
