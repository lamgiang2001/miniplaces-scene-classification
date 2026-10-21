"""
Plot training results. 

Makes two line plots:
  1. Validation accuracy (top-1 and top-5) per epoch (5 epochs)
  2. Loss (train and validation) per epoch (5 epochs)

"""
 
import argparse
import json
import os
import matplotlib.pyplot as plt
 
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="Model name used with train.py (e.g. mlp_baseline).")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--out_dir", default="figures")
    args = p.parse_args()
 
    path = os.path.join(args.results_dir, f"{args.model}_results.json")
    with open(path) as f:
        results = json.load(f)
 
    history = results["history"]
    epochs = [row["epoch"] for row in history]
    os.makedirs(args.out_dir, exist_ok=True)
 
    # Plot 1: validation accuracy per epoch
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, [row["val_top1"] for row in history], marker="o", label="Validation top-1")
    plt.plot(epochs, [row["val_top5"] for row in history], marker="o", label="Validation top-5")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{args.model}: validation accuracy per epoch")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    acc_path = os.path.join(args.out_dir, f"{args.model}_val_accuracy.png")
    plt.savefig(acc_path, dpi=200)
    plt.close()
 
    # Plot 2: loss per epoch
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, [row["train_loss"] for row in history], marker="o", label="Train loss")
    plt.plot(epochs, [row["val_loss"] for row in history], marker="o", label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title(f"{args.model}: loss per epoch")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    loss_path = os.path.join(args.out_dir, f"{args.model}_loss.png")
    plt.savefig(loss_path, dpi=200)
    plt.close()
 
    print(f"Saved {acc_path}")
    print(f"Saved {loss_path}")
 
 
if __name__ == "__main__":
    main()