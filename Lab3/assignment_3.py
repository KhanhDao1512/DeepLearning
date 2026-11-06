import os
import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from seqeval.metrics import f1_score

from tqdm import tqdm

from run_utils import start_logging, save_results
import numpy as np

from data_utils.phoner import PhoNER, Vocab, collate_fn, IGNORE_INDEX
from models.taggers.bilstm_encoder import BiLstmEncoder

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

DATA_DIR = "datasets/PhoNER_COVID19"
TRAIN_PATH = f"{DATA_DIR}/word/train_word.conll"
DEV_PATH = f"{DATA_DIR}/word/dev_word.conll"
TEST_PATH = f"{DATA_DIR}/word/test_word.conll"

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
            labels = item["labels"].to(device)              # (bs, len)

            # forward pass
            logits = model(input_ids, lengths)              # (bs, len, n_tags)
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss.append(loss.item())
            pbar.set_postfix({"loss": np.array(running_loss).mean()})
    return float(np.mean(running_loss))

@torch.no_grad()
def evaluate(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        epoch: int,
        vocab: Vocab
    ) -> float:
    model.eval()

    gts = []        # list các câu, mỗi câu là list nhãn thật
    pds = []        # list các câu, mỗi câu là list nhãn dự đoán
    with tqdm(dataloader, desc=f"Epoch {epoch} - Evaluating") as pbar:
        for item in pbar:
            input_ids = item["input_ids"].to(device)
            lengths = item["lengths"]
            labels = item["labels"]                         # (bs, len)

            logits = model(input_ids, lengths)
            predicted = logits.argmax(dim=-1).cpu()         # (bs, len)

            # bỏ phần padding, giữ đúng độ dài từng câu
            for gt, pd, n in zip(labels, predicted, lengths.tolist()):
                gts.append([vocab.i2l[i] for i in gt[:n].tolist()])
                pds.append([vocab.i2l[i] for i in pd[:n].tolist()])

    # F1 mức thực thể (entity-level, micro) theo seqeval
    f1 = f1_score(gts, pds)
    print(f"F1: {f1}")
    return f1

if __name__ == "__main__":
    start_logging("assignment_3")

    print("Loading vocab ...")
    vocab = Vocab(train_path=TRAIN_PATH)

    print("Creating datasets ...")
    train_dataset = PhoNER(path=TRAIN_PATH, vocab=vocab)
    dev_dataset = PhoNER(path=DEV_PATH, vocab=vocab)
    test_dataset = PhoNER(path=TEST_PATH, vocab=vocab)

    print("Creating dataloaders ...")
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    dev_dataloader = DataLoader(dev_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    print("Initializing model and utilities ...")
    model = BiLstmEncoder(
        vocab_size=vocab.len,
        hidden_size=256,
        n_layer=5,
        n_tags=vocab.n_labels,
        padding_idx=0
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX).to(device)

    epoch = 0
    best_f1 = 0
    best_epoch = 0
    patience = 0
    history = []
    ckpt_path = "checkpoints/assignment_3_best.pt"
    os.makedirs("checkpoints", exist_ok=True)
    while True:
        epoch += 1
        train_loss = train(model, train_dataloader, epoch, loss_fn, optimizer)
        f1 = evaluate(model, dev_dataloader, epoch, vocab)
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
    test_f1 = evaluate(model, test_dataloader, best_epoch, vocab)
    print(f"F1-score on test set: {test_f1}")

    save_results("assignment_3", {
        "best_epoch": best_epoch,
        "best_dev_f1": best_f1,
        "test_f1": test_f1,
        "history": history
    })
