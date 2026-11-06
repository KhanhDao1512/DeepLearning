import os 
import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from tqdm import tqdm

from run_utils import start_logging, save_results
import numpy as np

from data_utils.uit_vsfc import UIT_VSFC, Vocab, collate_fn
from models.classifiers.lstm import LstmModel

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def train(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        epoch: int,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer
    ):

    running_loss = []
    model.train()

    with tqdm(dataloader, desc=f"Epoch {epoch} - Training") as pbar:
        for item in pbar:
            input_ids = item["input_ids"].to(device)
            lengths = item["lengths"]
            labels = item["label"].to(device)
            
            # forward pass
            logits = model(input_ids, lengths)      # (bs, n_labels)
            loss = loss_fn(logits, labels)
            
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss.append(loss.item())
            pbar.set_postfix({
                "loss": np.array(running_loss).mean()
            })
    return float(np.mean(running_loss))

@torch.no_grad()
def evaluate(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        epoch: int
    ) -> float:
    model.eval()

    gts = []        # ground truths 
    pds = []        # predictions 
    with tqdm(dataloader, desc=f"Epoch {epoch} - Evaluating") as pbar:
        for item in pbar:
            input_ids = item["input_ids"].to(device)
            lengths = item["lengths"]
            labels = item["label"].to(device) # (bs, )
            
            # forward pass
            logits: torch.Tensor = model(input_ids, lengths)
            predicted_labels = logits.argmax(dim=-1) # (bs, )

            gts.extend(labels.tolist())
            pds.extend(predicted_labels.tolist())
    
    f1 = f1_score(gts, pds, average="macro")
    print(f"F1: {f1}")
    return f1

if __name__ == "__main__":
    start_logging("assignment_1")
    
    print("Loading vocab ...")
    vocab = Vocab(train_path="datasets/UIT-VSFC/UIT-VSFC-train.json")

    print("Creating datasets ...")
    train_dataset = UIT_VSFC(
        path="datasets/UIT-VSFC/UIT-VSFC-train.json",
        vocab=vocab
    )
    dev_dataset = UIT_VSFC(
        path="datasets/UIT-VSFC/UIT-VSFC-dev.json",
        vocab=vocab
    )
    test_dataset = UIT_VSFC(
        path="datasets/UIT-VSFC/UIT-VSFC-test.json",
        vocab=vocab
    )

    print("Creating dataloaders ...")
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size = 32,
        shuffle=True,
        collate_fn=collate_fn
    )
    dev_dataloader = DataLoader(
        dataset=dev_dataset,
        batch_size = 32,
        shuffle=True,
        collate_fn=collate_fn
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size = 32,
        shuffle=True,
        collate_fn=collate_fn
    )

    print("Initializing model and utilities ...")
    model = LstmModel(
        vocab_size=vocab.len,
        hidden_size=256,
        n_layer=5,
        n_labels=vocab.n_labels,
        padding_idx=0
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss().to(device)

    epoch = 0
    best_f1 = 0
    best_epoch = 0
    patience = 0
    history = []
    ckpt_path = "checkpoints/assignment_1_best.pt"
    os.makedirs("checkpoints", exist_ok=True)
    while True:
        epoch += 1
        train_loss = train(model, train_dataloader, epoch, loss_fn, optimizer)
        f1 = evaluate(model, dev_dataloader, epoch)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} dev_f1={f1:.4f}")
        history.append({"epoch": epoch, "train_loss": train_loss, "dev_f1": f1})

        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch
            patience = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience += 1

        if patience == 5:
            print("Patience reached! Training completed")
            break

    # đánh giá trên test bằng model tốt nhất theo dev
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"Loaded best model from epoch {best_epoch} (dev F1 = {best_f1:.4f})")
    test_f1 = evaluate(model, test_dataloader, best_epoch)
    print(f"F1-score on test set: {test_f1}")

    save_results("assignment_1", {
        "best_epoch": best_epoch,
        "best_dev_f1": best_f1,
        "test_f1": test_f1,
        "history": history
    })
