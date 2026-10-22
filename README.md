# miniplaces-scene-classification

## Train and evaluate the CNN

Install the Python dependencies in your active environment:

```bash
python -m pip install -r requirements.txt
```

Run a five-epoch CNN experiment:

```bash
python train.py --data_root "data 2" --image_root data/images --model cnn \
  --epochs 5 --batch_size 128 --train_subset 5000 --val_subset 2000 \
  --skip_test --num_workers 2 --out_dir results
```

The command reads labels from `data 2/data/train.txt`, uses a class-balanced
5,000-image subset for training, and measures top-1/top-5 accuracy on a separate
2,000-image validation subset drawn from the remaining training examples. The
official validation split is skipped for this experiment.

The shared trainer writes the epoch metrics table to `results/cnn_metrics.csv`,
the loss and accuracy plots to `results/cnn_metrics.svg`, the confusion-matrix
table and heatmap to `results/cnn_confusion_matrix.csv` and
`results/cnn_confusion_matrix.svg`, and the checkpoint and full metrics JSON
under `results/`. It also writes `results/cnn_validation_predictions.csv` with
the true class, predicted class, correct/incorrect flag, and confidence for each
evaluated validation image. The shared data loader applies the same image
preprocessing for training and evaluation.