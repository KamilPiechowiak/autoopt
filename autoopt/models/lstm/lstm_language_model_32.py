from torch import nn
import torch

from autoopt.models.lstm.lstm_language_model import LSTMLanguageModel


class LSTMLanguageModel32(LSTMLanguageModel):

    def __init__(self, vocab_size) -> None:
        super(LSTMLanguageModel32, self).__init__(vocab_size, 32)
