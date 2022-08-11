from torch import nn
import torch


class LSTMLanguageModel(nn.Module):

    def __init__(self, vocab_size, hidden_size: int = 128, num_layers: int = 2) -> None:
        super(LSTMLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x, _ = self.lstm(x)
        y = self.linear(x)
        return y.reshape(-1, y.shape[-1])
