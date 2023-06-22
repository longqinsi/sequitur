import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class DynamicRNN(nn.Module):
    """
    The wrapper version of recurrent modules including RNN, LSTM
    that support packed sequence batch.
    """

    def __init__(self, rnn_module):
        super().__init__()
        self.rnn_module = rnn_module

    def forward(self, x, len_x, initial_state=None):
        """
        Arguments
        ---------
        x : torch.FloatTensor
            padded input sequence tensor for RNN model
            Shape [batch_size, max_seq_len, embed_size]

        len_x : torch.LongTensor
            Length of sequences (b, )

        initial_state : torch.FloatTensor
            Initial (hidden, cell) states of RNN model.

        Returns
        -------
        A tuple of (padded_output, h_n) or (padded_output, (h_n, c_n))
            padded_output: torch.FloatTensor
                The output of all hidden for each elements. The hidden of padding elements will be assigned to
                a zero vector.
                Shape [batch_size, max_seq_len, hidden_size]

            h_n: torch.FloatTensor
                The hidden state of the last step for each packed sequence (not including padding elements)
                Shape [batch_size, hidden_size]
            c_n: torch.FloatTensor
                If rnn_model is RNN, c_n = None
                The cell state of the last step for each packed sequence (not including padding elements)
                Shape [batch_size, hidden_size]

        Example
        -------
        """
        # First sort the sequences in batch in the descending order of length
        sorted_len, idx = len_x.sort(dim=0, descending=True)
        sorted_x = x[idx]

        # Convert to packed sequence batch
        packed_x = pack_padded_sequence(sorted_x, lengths=sorted_len, batch_first=True)

        # Check init_state
        if initial_state is not None:
            if isinstance(initial_state, tuple):  # (h_0, c_0) in LSTM
                hx = [state[:, idx] for state in initial_state]
            else:
                hx = initial_state[:, idx]  # h_0 in RNN
        else:
            hx = None

        # Do forward pass
        self.rnn_module.flatten_parameters()
        packed_output, last_s = self.rnn_module(packed_x, hx)

        # pad the packed_output
        max_seq_len = x.size(1)
        padded_output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=max_seq_len)

        # Reverse to the original order
        _, reverse_idx = idx.sort(dim=0, descending=False)

        padded_output = padded_output[reverse_idx]

        if isinstance(self.rnn_module, nn.RNN):
            h_n, c_n = last_s[:, reverse_idx], None
        else:
            h_n, c_n = [s[:, reverse_idx] for s in last_s]

        return padded_output, (h_n, c_n)


if __name__ == '__main__':
    "A simple example to test"
    # prepare examples
    x = [torch.tensor([[1.0, 1.0],
                       [2.0, 2.0],
                       [3.0, 3.0],
                       [4.0, 4.0],
                       [5.0, 5.0]]),

         torch.tensor([[2.5, 2.5]]),

         torch.tensor([[2.2, 2.2],
                       [3.5, 3.5]])]
    len_x = [5, 1, 2]

    # pad the seq_batch
    padded_x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=0.01)
    """
    >>> padded_x
    tensor([[[1.0000, 1.0000],
             [2.0000, 2.0000],
             [3.0000, 3.0000],
             [4.0000, 4.0000],
             [5.0000, 5.0000]],
    
            [[2.5000, 2.5000],
             [0.0100, 0.0100],
             [0.0100, 0.0100],
             [0.0100, 0.0100],
             [0.0100, 0.0100]],
    
            [[2.2000, 2.2000],
             [3.5000, 3.5000],
             [0.0100, 0.0100],
             [0.0100, 0.0100],
             [0.0100, 0.0100]]])
    """

    # init 2 recurrent module: lstm, drnn
    rnn = nn.LSTM(input_size=2, hidden_size=3, bidirectional=True, batch_first=True)
    drnn = DynamicRNN(rnn)

    # get the outputs
    d_out, (dh_n, dc_n) = drnn(x, len_x)
    out, (h_n, c_n) = rnn(x)

    # compare two outputs
    print(d_out == out)
    """
    tensor([[[1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1]],
    
            [[1, 1, 1, 0, 0, 0], # only the forward direction is the same not the backward direction 
             [0, 0, 0, 0, 0, 0], 
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]],
    
            [[1, 1, 1, 0, 0, 0], # same as above
             [1, 1, 1, 0, 0, 0], # same as above
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]]], dtype=torch.uint8)
    """

    print(dh_n == h_n)
    """
    tensor([[[1, 1, 1], # since no padding in the first seq
             [0, 0, 0],
             [0, 0, 0]],
             
            [[1, 1, 1],
             [0, 0, 0],
             [0, 0, 0]]], dtype=torch.uint8)
    """

    print(dc_n == c_n)
    """
    tensor([[[1, 1, 1], # since no padding in the first seq
             [0, 0, 0],
             [0, 0, 0]],
             
            [[1, 1, 1],
             [0, 0, 0],
             [0, 0, 0]]], dtype=torch.uint8)
    """
