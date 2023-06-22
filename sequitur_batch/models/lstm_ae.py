# Third Party
import torch
import torch.nn as nn
from torch.nn.functional import pad
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence


############
# COMPONENTS
############


class Encoder(nn.Module):
    def __init__(self, input_dim, out_dim, h_dims, h_activ, out_activ, seq_lengths):
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.seq_lengths = seq_lengths

        layer_dims = [input_dim] + h_dims + [out_dim]
        self.num_layers = len(layer_dims) - 1
        self.layers = nn.ModuleList()
        for index in range(self.num_layers):
            layer = nn.LSTM(
                input_size=layer_dims[index],
                hidden_size=layer_dims[index + 1],
                num_layers=1,
                batch_first=True,
            )
            self.layers.append(layer)

        self.h_activ, self.out_activ = h_activ, out_activ

    def forward(self, x):
        for index, layer in enumerate(self.layers):
            x: PackedSequence = pack_padded_sequence(x, self.seq_lengths,
                                                     batch_first=True, enforce_sorted=False)
            x, (h_n, c_n) = layer(x)
            x, _ = pad_packed_sequence(x, batch_first=True)
            if self.h_activ and index < self.num_layers - 1:
                x = self.h_activ(x)
            elif self.out_activ and index == self.num_layers - 1:
                return self.out_activ(h_n).squeeze()

        return h_n.squeeze()


class Decoder(nn.Module):
    def __init__(self, input_dim, out_dim, h_dims, h_activ, seq_lengths):
        super(Decoder, self).__init__()

        self.seq_lengths = seq_lengths
        layer_dims = [input_dim] + h_dims + [h_dims[-1]]
        self.num_layers = len(layer_dims) - 1
        self.layers = nn.ModuleList()
        for index in range(self.num_layers):
            layer = nn.LSTM(
                input_size=layer_dims[index],
                hidden_size=layer_dims[index + 1],
                num_layers=1,
                batch_first=True,
            )
            self.layers.append(layer)

        self.h_activ = h_activ
        self.dense_matrix = nn.Parameter(
            torch.rand((layer_dims[-1], out_dim), dtype=torch.float), requires_grad=True
        )

    def forward(self, x):
        x0 = x
        seq_length_max = self.seq_lengths.max()
        x = torch.stack([pad(x[i].repeat(seq_len, 1), (0, 0, 0, seq_length_max - seq_len))
                         for i, seq_len in enumerate(self.seq_lengths)])
        for index, layer in enumerate(self.layers):
            x = pack_padded_sequence(x, self.seq_lengths, batch_first=True, enforce_sorted=False)
            x, _ = layer(x)
            x, _ = pad_packed_sequence(x, batch_first=True)

            if self.h_activ and index < self.num_layers - 1:
                x = self.h_activ(x)

        x1s = []
        for i in range(x.shape[0]):
            x1 = x[i, :self.seq_lengths[i], :]
            x1 = torch.mm(x1, self.dense_matrix)
            x1 = pad(x1, (0, 0, 0, seq_length_max - self.seq_lengths[i]))
            x1s.append(x1)
        return torch.stack(x1s)


######
# MAIN
######


class LSTM_AE(nn.Module):
    def __init__(
            self,
            input_dim,
            encoding_dim,
            seq_lengths,
            h_dims=[],
            h_activ=nn.Sigmoid(),
            out_activ=nn.Tanh(),
    ):
        super(LSTM_AE, self).__init__()

        self.encoder = Encoder(input_dim, encoding_dim, h_dims, h_activ, out_activ, seq_lengths)
        self.decoder = Decoder(encoding_dim, input_dim, h_dims[::-1], h_activ, seq_lengths)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
