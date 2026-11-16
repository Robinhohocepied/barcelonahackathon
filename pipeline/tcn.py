"""v4 sequence model: a small non-causal temporal conv net over per-frame inputs (25 fps), in the spirit of
Borghesi et al. (ESWA 2023). Training uses torch; inference is plain numpy (float64) so predict.py is exact and torch-free."""
import numpy as np

DIL = (1, 2, 4, 8, 16, 32); CH = 32; K = 3


def np_conv1d(x, w, b, d):
    """x (C_in, T), w (C_out, C_in, K), 'same' padding with dilation d"""
    C_out, C_in, k = w.shape; pad = d * (k - 1) // 2
    xp = np.pad(x, ((0, 0), (pad, pad)))
    out = np.zeros((C_out, x.shape[1])) + b[:, None]
    for j in range(k):
        out += w[:, :, j] @ xp[:, j * d: j * d + x.shape[1]]
    return out


def np_forward(params, F):
    """F (T, n_in) -> probabilities (T, 3) for H, A, D"""
    h = np_conv1d(F.T.astype(np.float64), params["inp.weight"], params["inp.bias"], 1)
    for i, d in enumerate(DIL):
        z = np.maximum(np_conv1d(h, params["blocks.%d.c1.weight" % i], params["blocks.%d.c1.bias" % i], d), 0)
        z = np_conv1d(z, params["blocks.%d.c2.weight" % i], params["blocks.%d.c2.bias" % i], 1)
        h = np.maximum(h + z, 0)
    o = np_conv1d(h, params["out.weight"], params["out.bias"], 1).T
    o = o - o.max(1, keepdims=True); e = np.exp(o)
    return e / e.sum(1, keepdims=True)


def torch_model(n_in):
    import torch.nn as nn
    class Block(nn.Module):
        def __init__(s, d):
            super().__init__(); s.c1 = nn.Conv1d(CH, CH, K, padding=d * (K - 1) // 2, dilation=d); s.c2 = nn.Conv1d(CH, CH, 1); s.dr = nn.Dropout(0.1)
        def forward(s, x):
            import torch
            return torch.relu(x + s.c2(s.dr(torch.relu(s.c1(x)))))
    class Net(nn.Module):
        def __init__(s):
            super().__init__(); s.inp = nn.Conv1d(n_in, CH, 1); s.blocks = nn.ModuleList([Block(d) for d in DIL]); s.out = nn.Conv1d(CH, 3, 1)
        def forward(s, x):
            h = s.inp(x)
            for b in s.blocks: h = b(h)
            return s.out(h)
    return Net()
