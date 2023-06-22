import torch
from torch import LongTensor
from torch.nn.functional import pad

from sequitur_batch.models import LSTM_AE
from sequitur_batch.quick_train import instantiate_model, train_model

if __name__ == '__main__':
    train_set = [torch.randn(10, 3).cuda() for _ in range(10)] + [torch.randn(3, 5).cuda() for _ in range(10)]

    encoder = train_model(
        src_train_set=train_set, verbose=True, lr=1e-3, epochs=2,
        clip_value=1, device=None, encoding_dim=7, h_dims=[64]
    )

    pass
    # x1 = torch.randn(10, 3).cuda()  # Sequence of 10 3D vectors
    # z1 = encoder(x1)  # z.shape = [7]
    # x1_prime = decoder(z1, seq_len=10)  # x_prime.shape = [10, 3]
    #
    # print(x1)
    # print(x1_prime)
    #
    # x2 = torch.randn(5, 3).cuda()  # Sequence of 10 3D vectors
    # z2 = encoder(x2)  # z.shape = [7]
    # x2_prime = decoder(z2, seq_len=5)  # x_prime.shape = [10, 3]
    #
    # print(x2)
    # print(x2_prime)
