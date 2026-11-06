import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BiLstmEncoder(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            hidden_size: int,
            n_layer: int,
            n_tags: int,
            padding_idx: int = 0,
            dropout: float = 0.2
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            padding_idx=padding_idx
        )

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=n_layer,
            bidirectional=True,
            dropout=dropout if n_layer > 1 else 0.0
        )

        self.dropout = nn.Dropout(dropout)

        # BiLSTM nối 2 chiều -> 2 * hidden
        self.classifier = nn.Linear(
            in_features=2 * hidden_size,
            out_features=n_tags
        )

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor):
        '''
            inputs: torch.LongTensor (bs, len)
            lengths: torch.LongTensor (bs,)
            output: logits (bs, len, n_tags)
        '''
        max_len = inputs.shape[1]
        embedded_feats = self.embedding(inputs)         # (bs, len, hidden)
        packed = pack_padded_sequence(embedded_feats, lengths.cpu(), batch_first=True, enforce_sorted=False)

        packed_out, _ = self.lstm(packed)
        # total_length để giữ độ dài bằng với nhãn đã pad trong batch
        features, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=max_len)  # (bs, len, 2*hidden)

        logits = self.classifier(self.dropout(features))    # (bs, len, n_tags)

        return logits
