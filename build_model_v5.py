"""v5 = sequence model trained on Metrica open (3) + SkillCorner (6) + IDSSE (7) (python train_v6.py all 30) on top of v4.1's
components; decoder chosen by ext_v6eval.py (two-source, normal + extra ball loss) -> ext/final_cfg_v5.txt."""
import ast, pickle, numpy as np
m = pickle.load(open("pipeline/model_v41.pkl", "rb")); z = np.load("ext/tcn6_all.npz"); m["tcn"] = {k: z[k] for k in z.files}
m["decode"] = ast.literal_eval(open("ext/final_cfg_v5.txt").read())
m["tcn_trained_on"] = ["metrica_sample_1-3", "skillcorner x6 (possession only)", "idsse x7"]
pickle.dump(m, open("pipeline/model_v5.pkl", "wb"))
