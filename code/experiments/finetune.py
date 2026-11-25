"""Does the protocol effect survive fine-tuning?

The frozen-encoder design is what makes the partition the only varying factor,
but it invites the obvious question: would a network free to adapt its features
close the gap? The mechanism says no — the gap comes from memorising
patient-specific staining, sectioning and acquisition, and a network that can
adapt has more capacity to memorise it, not less — so the prediction is that
fine-tuning widens the gap rather than closing it.

This runs the same matched contrast with the encoder unfrozen: identical folds,
identical seeds, identical schedule, the partition the only difference. It is
deliberately small (ResNet-18, whole images at 224 px, one seed) because it
exists to answer a yes/no question on CPU, not to chase a ceiling.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import protocols as P  # noqa: E402

# Pretrained weights carry subnormal floats; without this CPU convolutions run
# roughly thirty times slower.
torch.set_flush_denormal(True)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class Fields(Dataset):
    """Whole fields, resized. Fine-tuning over the full tile grid is not
    affordable on CPU and is not what the question needs."""

    def __init__(self, paths, labels, size=224, train=False):
        self.paths, self.labels, self.size, self.train = paths, labels, size, train

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = Image.open(self.paths[i]).convert("RGB").resize(
            (self.size, self.size), Image.BILINEAR)
        a = np.asarray(im, dtype=np.float32) / 255.0
        if self.train and np.random.rand() < 0.5:
            a = a[:, ::-1].copy()
        a = (a - MEAN) / STD
        return torch.from_numpy(a.transpose(2, 0, 1)), int(self.labels[i])


def _run_fold(idx, tr, te, classes, seed, epochs, lr, batch, size):
    torch.manual_seed(seed)
    np.random.seed(seed)
    cls = {c: i for i, c in enumerate(classes)}
    y = idx.superclass.map(cls).values
    paths = idx.path.values

    from torchvision.models import resnet18, ResNet18_Weights
    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Linear(net.fc.in_features, len(classes))
    net = net.to(memory_format=torch.channels_last).train()

    counts = np.bincount(y[tr], minlength=len(classes)).astype(np.float32)
    w = torch.tensor(counts.sum() / np.maximum(counts, 1), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=w / w.mean())
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)

    dl = DataLoader(Fields(paths[tr], y[tr], size, train=True),
                    batch_size=batch, shuffle=True, num_workers=2,
                    drop_last=len(tr) > batch)
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad(set_to_none=True)
            out = net(xb.to(memory_format=torch.channels_last))
            loss_fn(out, yb).backward()
            opt.step()

    net.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in DataLoader(Fields(paths[te], y[te], size),
                                batch_size=batch, num_workers=2):
            preds.append(net(xb.to(memory_format=torch.channels_last)).argmax(1).numpy())
    pred = np.concatenate(preds)
    return balanced_accuracy_score(y[te], pred), pred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path,
                    default=Path("results/tables/image_index.csv"))
    ap.add_argument("--out", type=Path,
                    default=Path("results/tables/finetune_folds.csv"))
    ap.add_argument("--n-splits", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    idx = pd.read_csv(a.index).reset_index(drop=True)
    classes = sorted(idx.superclass.unique())
    y = idx.superclass.values
    g = idx.patient_id.values

    rows, t0 = [], time.time()
    for seed in a.seeds:
        for protocol, groups in (("patient", g), ("image", None)):
            folds = P.make_folds(y, groups, a.n_splits, seed)
            for k, (tr, te) in enumerate(folds):
                ba, _ = _run_fold(idx, tr, te, classes, seed,
                                  a.epochs, a.lr, a.batch, a.size)
                rows.append(dict(model="resnet18-finetuned", task="super3",
                                 protocol=protocol, seed=seed, fold=k,
                                 balanced_accuracy=ba,
                                 n_train=len(tr), n_test=len(te),
                                 patient_overlap=P.check_disjoint(g[tr], g[te])))
                print(f"[{protocol}] seed {seed} fold {k}  bacc={ba:.3f}  "
                      f"({time.time() - t0:.0f}s elapsed)", flush=True)
                pd.DataFrame(rows).to_csv(a.out, index=False)
            done = [r for r in rows if r["protocol"] == protocol and r["seed"] == seed]
            print(f"  == {protocol}: {np.mean([r['balanced_accuracy'] for r in done]):.3f}",
                  flush=True)

    d = pd.DataFrame(rows)
    piv = d.pivot_table(index=["seed", "fold"], columns="protocol",
                        values="balanced_accuracy")
    print(f"\nfine-tuned inflation: {100 * (piv['image'] - piv['patient']).mean():+.1f} "
          f"percentage points over {len(piv)} matched folds")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
