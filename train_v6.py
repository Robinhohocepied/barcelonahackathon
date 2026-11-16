"""v5 sequence model. python train_v6.py <holdout spec> <epochs> ; holdout spec like 'metrica:1' or 'idsse:J03WPY+J03WMX' or 'all'.
Trains on native copies of: Metrica open (3), SkillCorner (6, possession only), IDSSE (all exported; real dead-ball flag)."""
import glob, os, sys, time, pickle, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "pipeline"))
import tcn
from train_v5 import chunks, SK6
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(2)
def held(spec, src, g):
    if spec == "all": return False
    s, ids = spec.split(":"); return src == s and g in ids.split("+")
if __name__ == "__main__":
    spec, EP = sys.argv[1], int(sys.argv[2]); tr = []
    for f in sorted(glob.glob(os.path.join(HERE, "ext", "p5_native_*.pkl"))):
        for r in pickle.load(open(f, "rb")):
            if (r["src"] == "sk" and r["game"] not in SK6) or held(spec, r["src"], r["game"]): continue
            y = r["label"].astype(np.int64).copy()
            if r["src"] == "sk": y[y == 2] = -100
            y[y == 3] = -100; tr.append(dict(F=r["F"], y=y))
    net = tcn.torch_model(tr[0]["F"].shape[1]); opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
    lossf = torch.nn.CrossEntropyLoss(ignore_index=-100); t0 = time.time()
    for ep in range(EP):
        net.train(); X, Y = chunks(tr); perm = torch.randperm(len(X)); tot = 0
        for i in range(0, len(X), 16):
            b = perm[i:i + 16]; opt.zero_grad(); l = lossf(net(X[b]), Y[b]); l.backward(); opt.step(); tot += l.item() * len(b)
    print(spec, "halves", len(tr), "final loss %.4f %.0fs" % (tot / len(X), time.time() - t0), flush=True)
    np.savez(os.path.join(HERE, "ext", "tcn6_%s.npz" % spec.replace(":", "_").replace("+", "-")), **{k: v.detach().numpy().astype(np.float64) for k, v in net.state_dict().items()})
