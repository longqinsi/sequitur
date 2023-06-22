# Standard Library
from statistics import mean

# Third Party
import torch
from torch import LongTensor
from torch.nn import MSELoss
from torch.nn.functional import pad

from sequitur_batch.models import LSTM_AE


###########
# UTILITIES
###########


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def instantiate_model(model, input_dim, encoding_dim, **kwargs):
    if model.__name__ in ("LINEAR_AE", "LSTM_AE"):
        return model(input_dim, encoding_dim, **kwargs)
    elif model.__name__ == "CONV_LSTM_AE":
        pass
        # if len(train_set[-1].shape) == 3:  # 2D elements
        #     return model(train_set[-1].shape[-2:], encoding_dim, **kwargs)
        # elif len(train_set[-1].shape) == 4:  # 3D elements
        #     return model(train_set[-1].shape[-3:], encoding_dim, **kwargs)


def train_model(
        src_train_set, verbose, lr, epochs, clip_value, encoding_dim,
        h_dims, device=None
):
    if device is None:
        device = get_device()

    input_dim = max([t.shape[1] for t in src_train_set])
    seq_lengths = LongTensor([t.shape[0] for t in src_train_set])
    max_seq_length = seq_lengths.max()
    train_set = torch.stack([pad(t, (0, input_dim - t.shape[1], 0, max_seq_length - t.shape[0]),
                                 mode='constant', value=0) for t in src_train_set]).to(device)
    seq_lengths = seq_lengths.cpu().numpy()

    model = LSTM_AE(input_dim=input_dim, encoding_dim=encoding_dim, seq_lengths=seq_lengths,
                    h_dims=h_dims)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = MSELoss(reduction="sum")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        # Forward pass
        train_set_prime = model(train_set)
        loss = criterion(train_set_prime, train_set)
        # Backward pass
        loss.backward()
        # Gradient clipping on norm
        if clip_value is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()
        if verbose:
            print(f"Epoch: {epoch}, Loss: {loss}")

    return model.encoder


def get_encodings(model, train_set, device=None):
    if device is None:
        device = get_device()
    model.eval()
    encodings = model.encoder(train_set.to(device))
    return encodings


######
# MAIN
######


def quick_train(
        model,
        train_set,
        encoding_dim,
        verbose=False,
        lr=1e-3,
        epochs=50,
        clip_value=1,
        denoise=False,
        device=None,
        **kwargs,
):
    model = instantiate_model(model, train_set, encoding_dim, **kwargs)
    losses = train_model(
        model, train_set, verbose, lr, epochs, clip_value, device
    )
    encodings = get_encodings(model, train_set, device)

    return model.encoder, model.decoder, encodings, losses
