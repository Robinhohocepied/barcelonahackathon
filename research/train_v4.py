"""Train the v4 TCN on external games. python research/train_v4.py <held-out metrica game or 'all'> [epochs]"""
import glob, os, sys, time, numpy as np, torch
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(HERE, "pipeline"))
import tcn
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(2)

def load():
    G = []
    for f in sorted(glob.glob(os.path.join(HERE, "ext", "f4_*.npz"))):
        z = np.load(f); src, g = os.path.basename(f)[3:-4].split("_")
        for k in range(2):
            y = z["y%d" % k].astype(np.int64).copy()
            if src == "sk": y[y == 2] = -100                      # SkillCorner has no real dead-ball flag: do not learn D from it
            y[y == 3] = -100
            G.append(dict(src=src, game=g, F=z["F%d" % k], y=y, ytrue=z["y%d" % k]))
    return G

def chunks(halves, L=2048):
    X, Y = [], []
    for h in halves:
        T = len(h["y"]); off = np.random.randint(0, L // 2)
        for s in range(off, T - L, L // 2):
            X.append(h["F"][s:s + L].T); Y.append(h["y"][s:s + L])
    return torch.tensor(np.array(X)), torch.tensor(np.array(Y))

if __name__ == "__main__":
    hold = sys.argv[1]; EP = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    G = load(); tr = [h for h in G if not (h["src"] == "metrica" and h["game"] == hold)]
    net = tcn.torch_model(G[0]["F"].shape[1]); opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
    lossf = torch.nn.CrossEntropyLoss(ignore_index=-100); t0 = time.time()
    for ep in range(EP):
        net.train(); X, Y = chunks(tr); perm = torch.randperm(len(X)); tot = 0
        for i in range(0, len(X), 16):
            b = perm[i:i + 16]; opt.zero_grad(); l = lossf(net(X[b]), Y[b]); l.backward(); opt.step(); tot += l.item() * len(b)
        if ep in (0, EP // 2, EP - 1): print("ep %d loss %.4f  %.0fs" % (ep, tot / len(X), time.time() - t0), flush=True)
    params = {k: v.detach().numpy().astype(np.float64) for k, v in net.state_dict().items()}
    np.savez(os.path.join(HERE, "ext", "tcn_%s.npz" % hold), **params)
    if hold != "all":
        te = [h for h in G if h["src"] == "metrica" and h["game"] == hold]
        out = {}
        for k, h in enumerate(te):
            out["P%d" % k] = tcn.np_forward(params, h["F"]); out["y%d" % k] = h["ytrue"]
        np.savez(os.path.join(HERE, "ext", "heldout_%s.npz" % hold), **out)
        acc = np.mean(np.concatenate([out["P%d" % k].argmax(1) == out["y%d" % k] for k in range(len(te))]))
        print("held-out game", hold, "frame accuracy %.3f" % acc, flush=True)
