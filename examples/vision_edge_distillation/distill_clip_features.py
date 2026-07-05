"""Feature-distill CLIP's vision capability into a compact CNN, on a GPU. Self-contained.

The honest edge path for vision the laptop couldn't reach: a compact student CNN learns to MIMIC CLIP's
512-d image embedding on CIFAR-10 (cosine feature distillation), then classifies zero-shot through
CLIP's FROZEN text head. So the deployed student carries CLIP's geometry in a few MB and needs no CLIP
at inference. Saves the student weights + a metrics JSON to retrieve.
"""

import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={DEV} {torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}", flush=True)

from datasets import load_dataset  # noqa: E402
from transformers import CLIPModel, CLIPProcessor  # noqa: E402

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).to(DEV).eval()
proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)

t0 = time.time()
tr = load_dataset("uoft-cs/cifar10", split="train")  # all 50k
te = load_dataset("uoft-cs/cifar10", split="test")  # all 10k
tr_imgs = [r["img"] for r in tr]
te_imgs = [r["img"] for r in te]
ytr = np.array([r["label"] for r in tr])
yte = np.array([r["label"] for r in te])
print(f"data loaded {len(tr_imgs)}+{len(te_imgs)} in {time.time() - t0:.0f}s", flush=True)


@torch.no_grad()
def clip_embed(imgs, bs=256):
    out = []
    for i in range(0, len(imgs), bs):
        inp = proc(images=imgs[i : i + bs], return_tensors="pt").to(DEV)
        vo = clip.vision_model(pixel_values=inp["pixel_values"])
        v = clip.visual_projection(vo.pooler_output)
        out.append(F.normalize(v, dim=-1).cpu())
    return torch.cat(out)


t0 = time.time()
etr = clip_embed(tr_imgs)  # teacher embeddings (the distillation target)
ete = clip_embed(te_imgs)
with torch.no_grad():
    tin = proc(text=[f"a photo of a {c}" for c in CLASSES], return_tensors="pt", padding=True).to(DEV)
    to = clip.text_model(input_ids=tin["input_ids"], attention_mask=tin["attention_mask"])
    tfeat = F.normalize(clip.text_projection(to.pooler_output), dim=-1).cpu()  # frozen zero-shot head
clip_zs = float(((ete @ tfeat.T).argmax(1).numpy() == yte).mean())
print(f"CLIP embeddings in {time.time() - t0:.0f}s | CLIP zero-shot acc {clip_zs:.4f}", flush=True)

# raw pixel tensors for the student
Xtr = torch.tensor(np.stack([np.array(i) for i in tr_imgs]), dtype=torch.float32).permute(0, 3, 1, 2).div(255)
Xte = torch.tensor(np.stack([np.array(i) for i in te_imgs]), dtype=torch.float32).permute(0, 3, 1, 2).div(255)
mean = Xtr.mean((0, 2, 3), keepdim=True)
std = Xtr.std((0, 2, 3), keepdim=True)
Xtr = ((Xtr - mean) / std).to(DEV)
Xte = ((Xte - mean) / std).to(DEV)
etr = etr.to(DEV)
tfeat = tfeat.to(DEV)


def block(ci, co, stride=1):
    return nn.Sequential(nn.Conv2d(ci, co, 3, stride, 1, bias=False), nn.BatchNorm2d(co), nn.ReLU())


class Student(nn.Module):
    def __init__(self, dim=512):
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

    def forward(self, x):
        return F.normalize(self.proj(self.body(x)), dim=-1)


student = Student().to(DEV)
npar = sum(p.numel() for p in student.parameters())
opt = torch.optim.AdamW(student.parameters(), lr=3e-3, weight_decay=5e-4)
EPOCHS = 40
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, epochs=EPOCHS, steps_per_epoch=(len(Xtr) + 255) // 256)


@torch.no_grad()
def evaluate():
    student.eval()
    preds = []
    for i in range(0, len(Xte), 512):
        preds.append((student(Xte[i : i + 512]) @ tfeat.T).argmax(1).cpu())
    p = torch.cat(preds).numpy()
    return float((p == yte).mean())


t0 = time.time()
for ep in range(EPOCHS):
    student.train()
    perm = torch.randperm(len(Xtr), device=DEV)
    for i in range(0, len(Xtr), 256):
        idx = perm[i : i + 256]
        opt.zero_grad()
        emb = student(Xtr[idx])
        loss = (1 - (emb * etr[idx]).sum(1)).mean()  # cosine feature-distillation loss
        loss.backward()
        opt.step()
        sched.step()
    if ep % 5 == 4 or ep == EPOCHS - 1:
        print(
            f"ep{ep + 1} loss={loss.item():.4f} student_zs_acc={evaluate():.4f} [{time.time() - t0:.0f}s]", flush=True
        )

acc = evaluate()
torch.save({k: v.cpu() for k, v in student.state_dict().items()}, "student.pt")
torch.save({"mean": mean, "std": std, "tfeat": tfeat.cpu()}, "student_head.pt")
metrics = {
    "clip_zero_shot_acc": clip_zs,
    "student_acc": acc,
    "retention": acc / clip_zs,
    "student_params": npar,
    "student_mb": round(npar * 4 / 1e6, 2),
    "epochs": EPOCHS,
    "train_seconds": round(time.time() - t0, 1),
    "device": str(DEV),
}
json.dump(metrics, open("metrics.json", "w"), indent=2)
print("RESULT " + json.dumps(metrics), flush=True)
