from pathlib import Path

import torch


def save_checkpoint(model, path, epoch, score, config, class_names):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "score": score,
            "model_state_dict": model.state_dict(),
            "config": config,
            "class_names": class_names,
        },
        path,
    )
