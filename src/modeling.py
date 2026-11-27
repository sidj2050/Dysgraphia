from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_NAMES = {"cnn": "CNN", "rnn": "RNN"}
MODEL_DESCRIPTIONS = {
    "cnn": "Convolutional neural network reading the handwriting as an image",
    "rnn": "Bidirectional GRU recurrent network reading the handwriting left-to-right",
}


# ---------------------------------------------------------------------- CNN
def _block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class DysgraphiaCNN(nn.Module):
    def __init__(self, channels=(8, 16, 32, 48), dropout=0.4):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.features = nn.Sequential(
            _block(1, c1), nn.MaxPool2d(2),
            _block(c1, c2), nn.MaxPool2d(2),
            _block(c2, c3), nn.MaxPool2d(2),
            _block(c3, c4), _block(c4, c4))
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * c4, 1)

    def forward(self, x):                          # x: (B, 1, H, W) in [0, 1]
        f = self.features(x)
        pooled = torch.cat([f.mean(dim=(2, 3)), f.amax(dim=(2, 3))], dim=1)
        return self.fc(self.drop(pooled)).squeeze(1)


# ---------------------------------------------------------------------- RNN
class DysgraphiaRNN(nn.Module):
    """Columns of the ink image are the time steps of a bidirectional GRU."""

    def __init__(self, in_height=32, embed=48, hidden=48, layers=2, dropout=0.3):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(in_height, embed), nn.LayerNorm(embed), nn.ReLU())
        self.gru = nn.GRU(embed, hidden, num_layers=layers, batch_first=True,
                          bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(4 * hidden, 1)

    @staticmethod
    def _lengths(seq):
        """Number of time steps up to the last column containing ink. seq: (B, T, H)."""
        occupied = seq.amax(dim=2) > 0.02          # (B, T)
        T = occupied.shape[1]
        idx = torch.arange(1, T + 1, device=seq.device).unsqueeze(0)
        return (occupied * idx).amax(dim=1).clamp(min=1)

    def forward(self, x):                          # x: (B, 1, H, W)
        x = F.avg_pool2d(x, 2).squeeze(1)          # (B, H/2, W/2)
        seq = x.transpose(1, 2)                    # (B, T, H/2): one column per step
        lengths = self._lengths(seq)
        h = self.embed(seq)
        packed = nn.utils.rnn.pack_padded_sequence(h, lengths.cpu(), batch_first=True,
                                                   enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=h.shape[1])
        valid = (torch.arange(out.shape[1], device=x.device).unsqueeze(0)
                 < lengths.unsqueeze(1)).unsqueeze(2)
        mean = (out * valid).sum(1) / valid.sum(1).clamp(min=1)
        mx = out.masked_fill(~valid, -1e4).amax(dim=1)
        return self.fc(self.drop(torch.cat([mean, mx], dim=1))).squeeze(1)


def make_model(name: str) -> nn.Module:
    if name == "cnn":
        return DysgraphiaCNN()
    if name == "rnn":
        return DysgraphiaRNN()
    raise ValueError(f"Unknown model '{name}' (expected 'cnn' or 'rnn')")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def canvas_to_tensor(canvas):
    """uint8 canvas(es) (H, W) or (N, H, W)  ->  float tensor (N, 1, H, W) in [0, 1]."""
    t = torch.as_tensor(canvas, dtype=torch.float32) / 255.0
    return t.unsqueeze(0).unsqueeze(0) if t.ndim == 2 else t.unsqueeze(1)
