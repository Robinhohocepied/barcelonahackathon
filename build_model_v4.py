"""Assemble pipeline/model_v4.pkl from model_v2.pkl, ext/tcn_all.npz (python train_v4.py all 25) and ext/final_cfg.txt (ext_v4ens.py on all held-out games)."""
import ast, pickle, numpy as np
v2 = pickle.load(open("pipeline/model_v2.pkl", "rb")); z = np.load("ext/tcn_all.npz")
m = dict(team=v2["team"], dead=v2["dead"], stack=v2["stack"], tcn={k: z[k] for k in z.files}, decode=ast.literal_eval(open("ext/final_cfg.txt").read()),
         trained_on=v2["trained_on"], stack_trained_on=v2["stack_trained_on"],
         tcn_trained_on=["metrica_sample_1", "metrica_sample_2", "metrica_sample_3", "sk1886347", "sk1996435", "sk2017461", "sk2016236", "sk2006229", "sk1874553"])
pickle.dump(m, open("pipeline/model_v4.pkl", "wb"))
