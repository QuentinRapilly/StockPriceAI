import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import Transformer
from torch.utils.data import dataset


class TransformerStock(nn.Module):

    def __init__(self, nout: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nout = nout

        self.transformer = Transformer(d_model = d_model, nhead= nhead, num_encoder_layers=nlayers, num_decoder_layers=6,
        dim_feedforward=nout, batch_first=True, dropout= dropout)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)



