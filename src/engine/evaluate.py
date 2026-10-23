import sys

import torch
from tqdm import tqdm

from .metrics import calculate_metrics


def evaluate(model, loader, loss_fn, device, epoch=None, total_epochs=None):
    model.eval()
    all_labels = []
    all_predictions = []
    total_loss = 0.0
    total_items = 0

    with torch.inference_mode():
        for images, labels in tqdm(
            loader,
            desc="Evaluate",
            leave=False,
            disable=not sys.stderr.isatty(),
        ):
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = loss_fn(logits, labels)
            predictions = logits.argmax(dim=1)

            total_loss += loss.item() * labels.size(0)
            total_items += labels.size(0)
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    metrics = calculate_metrics(all_labels, all_predictions)
    metrics["loss"] = total_loss / max(total_items, 1)
    return metrics
