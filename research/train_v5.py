"""v5 TCN. python research/train_v5.py <held-out metrica game|all> <epochs> <variants e.g. native,aug> <sk: 6|20>"""
import glob, os, sys, time, pickle, numpy as np, torch
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(HERE, "pipeline"))
import tcn
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(2)
SK6 = {"1886347", "1996435", "2017461", "2016236", "2006229", "1874553"}
def load(variants, nsk):
    H = []
    for v in variants:
        for f in sorted(glob.glob(os.path.join(HERE, "ext", "p5_%s_*.pkl" % v))):
            for r in pickle.load(open(f, "rb")):
                if r["src"] == "sk" and nsk == 6 and r["game"] not in SK6: continue
                y = r["label"].astype(np.int64).copy()
                if r["src"] == "sk": y[y == 2] = -100
                y[y == 3] = -100
                H.append(dict(src=r["src"], game=r["game"], variant=v, F=r["F"], y=y))
    return H
def chunks(halves, L=2048, cap=650):
    X, Y = [], []
    for h in halves:
        T = len(h["y"]); off = np.random.randint(0, L // 2)
        for s in range(off, T - L, L // 2): X.append((h, s))
    sel = np.random.permutation(len(X))[:cap]
    return torch.tensor(np.array([X[i][0]["F"][X[i][1]:X[i][1] + L].T for i in sel])), torch.tensor(np.array([X[i][0]["y"][X[i][1]:X[i][1] + L] for i in sel]))
if __name__ == "__main__":
    hold, EP, variants, nsk = sys.argv[1], int(sys.argv[2]), sys.argv[3].split(","), int(sys.argv[4])
    tag = "%s_%s_sk%d" % ("+".join(variants), hold, nsk)
    tr = [h for h in load(variants, nsk) if not (h["src"] == "metrica" and h["game"] == hold)]
    net = tcn.torch_model(tr[0]["F"].shape[1]); opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
    lossf = torch.nn.CrossEntropyLoss(ignore_index=-100); t0 = time.time()
    for ep in range(EP):
        net.train(); X, Y = chunks(tr); perm = torch.randperm(len(X)); tot = 0
        for i in range(0, len(X), 16):
            b = perm[i:i + 16]; opt.zero_grad(); l = lossf(net(X[b]), Y[b]); l.backward(); opt.step(); tot += l.item() * len(b)
        if ep in (0, EP - 1): print(tag, "ep %d loss %.4f %.0fs" % (ep, tot / len(X), time.time() - t0), flush=True)
    np.savez(os.path.join(HERE, "ext", "tcn5_%s.npz" % tag), **{k: v.detach().numpy().astype(np.float64) for k, v in net.state_dict().items()})
