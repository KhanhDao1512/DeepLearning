import csv
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn, optim

from src.data_utils import build_loaders
from src.engine.checkpoint import save_checkpoint
from src.engine.evaluate import evaluate
from src.engine.train import train_one_epoch
from src.models import create_model


def run_assignment(config_path):
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    training_config = config["training"]
    seed = int(training_config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, eval_loader, num_classes, class_names = build_loaders(
        config, model_name=config["model"]["name"]
    )
    model = create_model(config, num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss()
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = optim.Adam(
        trainable_parameters,
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config.get("weight_decay", 0.0)),
    )

    output_config = config["output"]
    checkpoint_path = Path(output_config["checkpoint_dir"]) / "best_model.pth"
    metrics_path = Path(output_config["metrics_file"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    best_score = -1.0
    epochs = int(training_config["epochs"])
    start_epoch = 1
    resume = bool(training_config.get("resume", False))

    if resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        best_score = float(checkpoint.get("score", best_score))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1

    if start_epoch > epochs:
        print(f"Already complete: {start_epoch - 1}/{epochs} epochs", flush=True)
        print(f"Checkpoint: {checkpoint_path}", flush=True)
        print(f"Metrics: {metrics_path}", flush=True)
        return

    file_mode = "a" if resume and metrics_path.exists() else "w"
    with metrics_path.open(file_mode, newline="", encoding="utf-8") as file:
        fields = ["epoch", "train_loss", "eval_loss", "accuracy", "precision", "recall", "f1_macro"]
        writer = csv.DictWriter(file, fieldnames=fields)
        if file_mode == "w":
            writer.writeheader()

        for epoch in range(start_epoch, epochs + 1):
            train_loss = train_one_epoch(
                model, train_loader, loss_fn, optimizer, device, epoch, epochs
            )
            metrics = evaluate(model, eval_loader, loss_fn, device)
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "eval_loss": metrics.pop("loss"),
                **metrics,
            }
            writer.writerow(row)
            file.flush()

            printable = " ".join(
                f"{key}={value:.4f}" for key, value in row.items() if key != "epoch"
            )
            print(f"Epoch {epoch}/{epochs}: {printable}", flush=True)

            metric_name = training_config.get("selection_metric", "f1_macro")
            score = float(metrics[metric_name])
            if score > best_score:
                best_score = score
                save_checkpoint(model, checkpoint_path, epoch, score, config, class_names)

    print(f"Best {training_config.get('selection_metric', 'f1_macro')}: {best_score:.4f}", flush=True)
    print(f"Checkpoint: {checkpoint_path}", flush=True)
    print(f"Metrics: {metrics_path}", flush=True)
