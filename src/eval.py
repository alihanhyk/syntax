import dill
import numpy as np
import pandas as pd

N = 25
H200 = 200 - 2 * N
H400 = 400 - 2 * N

algs = list()
algs.append(('banuni', 'Conventional study'))
algs.append(('banada', 'Thresholding bandits'))
algs.append(('synuni', 'Synthetic study'))
algs.append(('synsta', 'Synthetic design'))
algs.append(('synada', 'SYNTAX'))

metrics = pd.MultiIndex.from_product([
    ['Diminishing', 'Increasing'],
    ['H = 200', 'H = 400'],
    ['FPR', 'TPR']])

df = pd.DataFrame(index=[alg_name for _, alg_name in algs], columns=metrics)

for alg_tag, alg in algs:

    with open(f'res/dim-{alg_tag}.obj', 'rb') as file:
        dim_res, _, dim_truth = dill.load(file)
        dim_h200 = (dim_res[:,:,H200-1,:] > 0.)
        dim_h400 = (dim_res[:,:,H400-1,:] > 0.)
        dim_truth = dim_truth.astype(bool)
        
    with open(f'res/inc-{alg_tag}.obj', 'rb') as file:
        inc_res, _, inc_truth = dill.load(file)
        inc_h200 = (inc_res[:,:,H200-1,:] > 0.)
        inc_h400 = (inc_res[:,:,H400-1,:] > 0.)
        inc_truth = inc_truth.astype(bool)

    FPR_dim_h200 = np.mean((dim_h200 & ~dim_truth).sum(-1) / (~dim_truth).sum(-1), axis=-1)
    FPR_dim_h400 = np.mean((dim_h400 & ~dim_truth).sum(-1) / (~dim_truth).sum(-1), axis=-1)
    FPR_inc_h200 = np.mean((inc_h200 & ~inc_truth).sum(-1) / (~inc_truth).sum(-1), axis=-1)
    FPR_inc_h400 = np.mean((inc_h400 & ~inc_truth).sum(-1) / (~inc_truth).sum(-1), axis=-1)

    TPR_dim_h200 = np.mean((dim_h200 & dim_truth).sum(-1) / (dim_truth).sum(-1), axis=-1)
    TPR_dim_h400 = np.mean((dim_h400 & dim_truth).sum(-1) / (dim_truth).sum(-1), axis=-1)
    TPR_inc_h200 = np.mean((inc_h200 & inc_truth).sum(-1) / (inc_truth).sum(-1), axis=-1)
    TPR_inc_h400 = np.mean((inc_h400 & inc_truth).sum(-1) / (inc_truth).sum(-1), axis=-1)

    df.loc[alg, ("Diminishing", "H = 200", "FPR")] = f"{FPR_dim_h200.mean()*100:.1f}% ({FPR_dim_h200.std()*100:.1f}%)"
    df.loc[alg, ("Diminishing", "H = 200", "TPR")] = f"{TPR_dim_h200.mean()*100:.1f}% ({TPR_dim_h200.std()*100:.1f}%)"
    df.loc[alg, ("Diminishing", "H = 400", "FPR")] = f"{FPR_dim_h400.mean()*100:.1f}% ({FPR_dim_h400.std()*100:.1f}%)"
    df.loc[alg, ("Diminishing", "H = 400", "TPR")] = f"{TPR_dim_h400.mean()*100:.1f}% ({TPR_dim_h400.std()*100:.1f}%)"

    df.loc[alg, ("Increasing", "H = 200", "FPR")] = f"{FPR_inc_h200.mean()*100:.1f}% ({FPR_inc_h200.std()*100:.1f}%)"
    df.loc[alg, ("Increasing", "H = 200", "TPR")] = f"{TPR_inc_h200.mean()*100:.1f}% ({TPR_inc_h200.std()*100:.1f}%)"
    df.loc[alg, ("Increasing", "H = 400", "FPR")] = f"{FPR_inc_h400.mean()*100:.1f}% ({FPR_inc_h400.std()*100:.1f}%)"
    df.loc[alg, ("Increasing", "H = 400", "TPR")] = f"{TPR_inc_h400.mean()*100:.1f}% ({TPR_inc_h400.std()*100:.1f}%)"

print(df)
