"""
Error analysis for a trained model on the held-out TEST set (official val/).

Run this after training, once the model and hyperparameters are final: it is
post-hoc analysis, and nothing here should be used to tune the model. Writes:
  <figures_dir>/<model>_confusion.png   100 x 100 confusion matrix (image counts)
  <figures_dir>/<model>_successes.png   the most confident correct predictions
  <figures_dir>/<model>_failures.png    a random sample of misclassified images
  <checkpoint folder>/<model>_error_analysis.json
                                        per-class accuracy, best/worst classes,
                                        most confused class pairs

Example (run from the repo root):
  python error_analysis.py --data_root /path/to/miniplaces --model mlp_baseline \
      --checkpoint results/lr1e-4/mlp_baseline_best.pt
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from data_pipeline import build_dataloaders
from train import build_model

NUM_CLASSES = 100


def load_class_names(data_root):
    """
    categories.txt lines look like '/a/abbey 0' or '/c/church/outdoor 12'.
    Drop the leading letter folder, keep the rest ('abbey', 'church/outdoor'):
    several classes share a last component such as 'outdoor'.
    """
    names = {}
    with open(os.path.join(data_root, "data", "categories.txt")) as f:
        for line in f:
            line = line.strip()
            if line:
                path, idx = line.rsplit(" ", 1)
                names[int(idx)] = path.lstrip("/").split("/", 1)[-1]
    class_names = [names[i] for i in range(NUM_CLASSES)]
    assert len(set(class_names)) == NUM_CLASSES, "Class names are not unique."
    return class_names


@torch.no_grad()
def predict(model, loader, device):
    """Labels, top-5 predicted classes, and top-1 probability for every image, in loader order."""
    model.eval()
    labels, top5, conf = [], [], []
    for images, y in loader:
        probs = model(images.to(device)).softmax(dim=1)
        top_probs, top_idx = probs.topk(5, dim=1)
        labels.append(y)
        top5.append(top_idx.cpu())
        conf.append(top_probs[:, 0].cpu())
    return torch.cat(labels).numpy(), torch.cat(top5).numpy(), torch.cat(conf).numpy()


def image_grid(dataset, names, indices, preds, labels, conf, title, path, cols=4):
    rows = -(-len(indices) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.6 * rows))
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, indices):
        rel_path, _ = dataset.pairs[i]
        image = Image.open(os.path.join(dataset.image_root, rel_path)).convert("RGB")
        ax.imshow(image.resize((128, 128)))
        ax.set_title(f"true: {names[labels[i]]}\npred: {names[preds[i]]} ({conf[i]:.2f})",
                     fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("Saved", path)


def main():
    p = argparse.ArgumentParser(description="Per-class accuracy, confusion matrix, and example images.")
    p.add_argument("--data_root", required=True,
                   help="Folder containing data/train.txt, data/val.txt, data/categories.txt.")
    p.add_argument("--image_root", default=None,
                   help="Folder the label paths are relative to (same meaning as in train.py).")
    p.add_argument("--model", required=True, help="Same value that was passed to train.py --model.")
    p.add_argument("--checkpoint", required=True,
                   help="Best checkpoint saved by train.py (<out_dir>/<model>_best.pt).")
    p.add_argument("--figures_dir", default="figures")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42,
                   help="Seed for the random sample of failure examples.")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    _, _, test_loader = build_dataloaders(
        args.data_root, batch_size=args.batch_size, num_workers=args.num_workers,
        image_root=args.image_root)
    names = load_class_names(args.data_root)

    model = build_model(args.model).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    labels, top5, conf = predict(model, test_loader, device)
    preds = top5[:, 0]

    top1_acc = float((preds == labels).mean())
    top5_acc = float((top5 == labels[:, None]).any(axis=1).mean())
    print(f"Test top-1 {top1_acc:.3f} | top-5 {top5_acc:.3f}  ({len(labels)} images)")

    # Traceability check: the numbers should match what train.py saved for this checkpoint.
    ckpt_dir = os.path.dirname(args.checkpoint) or "."
    results_path = os.path.join(ckpt_dir, f"{args.model}_results.json")
    if os.path.isfile(results_path):
        with open(results_path) as f:
            saved = json.load(f)["test"]
        ok = abs(saved["top1"] - top1_acc) < 1e-3 and abs(saved["top5"] - top5_acc) < 1e-3
        print(f"Saved test result ({results_path}): top-1 {saved['top1']:.3f}, "
              f"top-5 {saved['top5']:.3f} -> {'matches' if ok else 'DIFFERS'}")

    # Confusion matrix (rows: true class, columns: predicted class) and per-class accuracy.
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    np.add.at(cm, (labels, preds), 1)
    support = cm.sum(axis=1)
    per_class = np.divide(np.diag(cm), support, out=np.zeros(NUM_CLASSES), where=support > 0)
    order = np.argsort(per_class, kind="stable")

    def ranked(indices):
        return [{"class": names[i], "accuracy": round(float(per_class[i]), 3)} for i in indices]

    best, worst = ranked(order[::-1][:10]), ranked(order[:10])

    off_diagonal = cm.copy()
    np.fill_diagonal(off_diagonal, 0)
    confusions = []
    for k in np.argsort(off_diagonal, axis=None, kind="stable")[::-1][:10]:
        true_idx, pred_idx = divmod(int(k), NUM_CLASSES)
        confusions.append({"true": names[true_idx], "predicted": names[pred_idx],
                           "count": int(off_diagonal[true_idx, pred_idx])})

    print("\nBest classes :", ", ".join(f"{c['class']} ({c['accuracy']:.2f})" for c in best))
    print("Worst classes:", ", ".join(f"{c['class']} ({c['accuracy']:.2f})" for c in worst))
    print("Most confused (true -> predicted):")
    for c in confusions:
        print(f"  {c['true']} -> {c['predicted']}: {c['count']} images")

    analysis_path = os.path.join(ckpt_dir, f"{args.model}_error_analysis.json")
    with open(analysis_path, "w") as f:
        json.dump({"checkpoint": args.checkpoint, "split": "test (official val/)",
                   "n_images": int(len(labels)), "top1": top1_acc, "top5": top5_acc,
                   "best_classes": best, "worst_classes": worst, "top_confusions": confusions,
                   "per_class_accuracy": {names[i]: round(float(per_class[i]), 3)
                                          for i in range(NUM_CLASSES)}}, f, indent=2)
    print("Saved", analysis_path)

    # Figures
    os.makedirs(args.figures_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted class index")
    ax.set_ylabel("True class index")
    ax.set_title(f"{args.model}: confusion matrix (test set, image counts)")
    fig.colorbar(image, ax=ax, label="Images")
    fig.tight_layout()
    confusion_path = os.path.join(args.figures_dir, f"{args.model}_confusion.png")
    fig.savefig(confusion_path, dpi=200)
    plt.close(fig)
    print("Saved", confusion_path)

    dataset = test_loader.dataset
    correct = np.where(preds == labels)[0]
    wrong = np.where(preds != labels)[0]
    successes = correct[np.argsort(-conf[correct], kind="stable")][:12]
    rng = np.random.default_rng(args.seed)
    failures = rng.choice(wrong, size=min(12, len(wrong)), replace=False)
    image_grid(dataset, names, successes, preds, labels, conf,
               f"{args.model}: most confident correct predictions (test set)",
               os.path.join(args.figures_dir, f"{args.model}_successes.png"))
    image_grid(dataset, names, failures, preds, labels, conf,
               f"{args.model}: random misclassified images (test set)",
               os.path.join(args.figures_dir, f"{args.model}_failures.png"))


if __name__ == "__main__":
    main()
