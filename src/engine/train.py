import sys

import torch
from tqdm import tqdm


def train_one_epoch(model, loader, loss_fn, optimizer, device, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    total_items = 0

    progress = tqdm(
        loader,
        desc=f"Epoch {epoch}/{total_epochs} train",
        leave=False,
        disable=not sys.stderr.isatty(),
    )
    for images, labels in progress:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_items += batch_size
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(total_items, 1)
