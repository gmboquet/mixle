"""Feature-distill CLIP's vision capability into a compact CNN on a GPU -- the producing half of a
two-machine round trip.

What it demonstrates: instead of distilling CLIP's *labels*, a small CNN is trained with a cosine loss
to reproduce CLIP's normalized image EMBEDDING for each CIFAR-10 image. The student therefore inherits
CLIP's feature geometry, and can then classify zero-shot through CLIP's *frozen text head* -- which is
saved alongside it. At inference the student never touches CLIP.

Takeaway: what makes a student portable is matching the teacher's representation, not its decisions.
Copying decisions gives you a classifier for one label set; copying the embedding gives you something
that still works through a text head, on a machine that has no CLIP and no GPU. (The contrast case --
distilling from raw pixels instead of from a good representation, which loses most of the teacher --
is in ``examples/foundation_to_edge.py`` section 2b.)

Writes three artifacts into ``--output-dir``: ``student.pt`` (the CNN weights), ``student_head.pt``
(the frozen zero-shot text head), and ``metrics.json`` (measurements plus the pinned model/dataset
revisions and dataset fingerprints). ``verify_on_laptop.py`` consumes all three on CPU, authenticating
each by SHA-256 -- obtain those digests over the same trusted channel as the release evidence.

Any accuracy this prints is a measurement of THAT run on THAT hardware and dependency set, not a
release claim; see this directory's README.md.

Requires a CUDA GPU, network access, and ``pip install torch transformers datasets pillow``.
Importing this module defines the student only. Network access, dataset materialization, training, and
artifact writes happen exclusively in ``main()``.

Run: ``python examples/vision_edge_distillation/distill_clip_features.py [--output-dir DIR]``
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
CLIP_ID = "openai/clip-vit-base-patch32"
CLIP_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
CIFAR10_ID = "uoft-cs/cifar10"
CIFAR10_REVISION = "0b2714987fa478483af9968de7c934580d0bb9a2"


def block(ci: int, co: int, stride: int = 1) -> nn.Sequential:
    """One convolution/batch-normalization/ReLU student block."""
    return nn.Sequential(nn.Conv2d(ci, co, 3, stride, 1, bias=False), nn.BatchNorm2d(co), nn.ReLU())


class Student(nn.Module):
    """Compact image encoder trained to reproduce CLIP embedding geometry."""

    def __init__(self, dim: int = 512) -> None:
        super().__init__()
        self.body = nn.Sequential(
            block(3, 64),
            block(64, 64),
            nn.MaxPool2d(2),
            block(64, 128),
            block(128, 128),
            nn.MaxPool2d(2),
            block(128, 256),
            block(256, 256),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.proj = nn.Linear(256, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(self.body(x)), dim=-1)


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="directory for student.pt, student_head.pt, and metrics.json",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the explicit network/GPU training workflow and write its receipt."""
    args = _arguments(argv)
    from datasets import load_dataset
    from transformers import CLIPModel, CLIPProcessor

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    print(f"device={device} {device_name}", flush=True)

    clip = CLIPModel.from_pretrained(CLIP_ID, revision=CLIP_REVISION, use_safetensors=True).to(device).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_ID, revision=CLIP_REVISION, use_fast=True)

    started = time.time()
    train = load_dataset(CIFAR10_ID, split="train", revision=CIFAR10_REVISION)
    test = load_dataset(CIFAR10_ID, split="test", revision=CIFAR10_REVISION)
    train_images = [record["img"] for record in train]
    test_images = [record["img"] for record in test]
    test_labels = np.array([record["label"] for record in test])
    print(f"data loaded {len(train_images)}+{len(test_images)} in {time.time() - started:.0f}s", flush=True)

    @torch.no_grad()
    def clip_embed(images, batch_size: int = 256) -> torch.Tensor:
        output = []
        for offset in range(0, len(images), batch_size):
            inputs = processor(images=images[offset : offset + batch_size], return_tensors="pt").to(device)
            vision = clip.vision_model(pixel_values=inputs["pixel_values"])
            output.append(F.normalize(clip.visual_projection(vision.pooler_output), dim=-1).cpu())
        return torch.cat(output)

    started = time.time()
    train_embeddings = clip_embed(train_images)
    test_embeddings = clip_embed(test_images)
    with torch.no_grad():
        text_inputs = processor(
            text=[f"a photo of a {category}" for category in CLASSES],
            return_tensors="pt",
            padding=True,
        ).to(device)
        text = clip.text_model(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
        )
        text_features = F.normalize(clip.text_projection(text.pooler_output), dim=-1).cpu()
    clip_accuracy = float(((test_embeddings @ text_features.T).argmax(1).numpy() == test_labels).mean())
    print(
        f"CLIP embeddings in {time.time() - started:.0f}s | CLIP zero-shot acc {clip_accuracy:.4f}",
        flush=True,
    )

    x_train = (
        torch.tensor(np.stack([np.array(image) for image in train_images]), dtype=torch.float32)
        .permute(0, 3, 1, 2)
        .div(255)
    )
    x_test = (
        torch.tensor(np.stack([np.array(image) for image in test_images]), dtype=torch.float32)
        .permute(0, 3, 1, 2)
        .div(255)
    )
    mean = x_train.mean((0, 2, 3), keepdim=True)
    std = x_train.std((0, 2, 3), keepdim=True)
    x_train = ((x_train - mean) / std).to(device)
    x_test = ((x_test - mean) / std).to(device)
    train_embeddings = train_embeddings.to(device)
    text_features = text_features.to(device)

    student = Student().to(device)
    parameter_count = sum(parameter.numel() for parameter in student.parameters())
    optimizer = torch.optim.AdamW(student.parameters(), lr=3e-3, weight_decay=5e-4)
    epochs = 40
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        3e-3,
        epochs=epochs,
        steps_per_epoch=(len(x_train) + 255) // 256,
    )

    @torch.no_grad()
    def evaluate() -> float:
        student.eval()
        predictions = []
        for offset in range(0, len(x_test), 512):
            predictions.append((student(x_test[offset : offset + 512]) @ text_features.T).argmax(1).cpu())
        return float((torch.cat(predictions).numpy() == test_labels).mean())

    started = time.time()
    for epoch in range(epochs):
        student.train()
        permutation = torch.randperm(len(x_train), device=device)
        for offset in range(0, len(x_train), 256):
            indices = permutation[offset : offset + 256]
            optimizer.zero_grad()
            embedding = student(x_train[indices])
            loss = (1 - (embedding * train_embeddings[indices]).sum(1)).mean()
            loss.backward()
            optimizer.step()
            schedule.step()
        if epoch % 5 == 4 or epoch == epochs - 1:
            print(
                f"ep{epoch + 1} loss={loss.item():.4f} student_zs_acc={evaluate():.4f} [{time.time() - started:.0f}s]",
                flush=True,
            )

    accuracy = evaluate()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({name: value.cpu() for name, value in student.state_dict().items()}, args.output_dir / "student.pt")
    torch.save(
        {"mean": mean, "std": std, "tfeat": text_features.cpu()},
        args.output_dir / "student_head.pt",
    )
    metrics = {
        "assets": {
            "clip": {"repository": CLIP_ID, "revision": CLIP_REVISION},
            "cifar10": {
                "repository": CIFAR10_ID,
                "revision": CIFAR10_REVISION,
                "train_fingerprint": train._fingerprint,
                "test_fingerprint": test._fingerprint,
            },
        },
        "clip_zero_shot_acc": clip_accuracy,
        "student_acc": accuracy,
        "retention": accuracy / clip_accuracy,
        "student_params": parameter_count,
        "student_mb": round(parameter_count * 4 / 1e6, 2),
        "epochs": epochs,
        "train_seconds": round(time.time() - started, 1),
        "device": str(device),
    }
    metrics_path = args.output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")
    print("RESULT " + json.dumps(metrics), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
